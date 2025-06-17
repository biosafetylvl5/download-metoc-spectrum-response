#!/usr/bin/env python3
"""
SRF File Downloader.

A simple script to download SRF files based on extracted YAML data.
Supports downloading by platform, instrument, specific channel, or all files.
"""

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Optional, Optional

import aiohttp
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Initialize Rich console
console = Console()
app = typer.Typer(help="Download SRF files from extracted YAML data")

# Constants
HTTP_OK = 200
DEFAULT_OUTPUT_DIR = "srf_downloads"
DEFAULT_MAX_CONCURRENT = 5

# Typer defaults - defined at module level to avoid B008
YAML_FILE_ARG = typer.Argument(None, help="Path to SRF YAML file (uses default if not provided)")
OUTPUT_DIR_OPT = typer.Option(Path(DEFAULT_OUTPUT_DIR), "--output", "-o", help="Output directory")
PLATFORM_OPT = typer.Option(None, "--platform", "-p", help="Filter by platform")
INSTRUMENT_OPT = typer.Option(
    None, "--instrument", "-i", help="Filter by instrument"
)
CHANNEL_OPT = typer.Option(None, "--channel", "-c", help="Filter by channel")
FILE_TYPE_OPT = typer.Option(None, "--type", "-t", help="Filter by file type")
FILE_TYPE_SEARCH_OPT = typer.Option(
    None, "--type", "-t", help="Filter by file type (txt, tar.gz, flt)"
)
MAX_CONCURRENT_OPT = typer.Option(
    DEFAULT_MAX_CONCURRENT, "--concurrent", help="Maximum concurrent downloads"
)
DRY_RUN_OPT = typer.Option(
    False, "--dry-run", help="Show what would be downloaded without downloading"
)


class SRFFileError(Exception):
    """Base exception for SRF file operations."""


class YAMLFileError(SRFFileError):
    """Exception for YAML file operations."""


class FileNotFoundError(SRFFileError):  # noqa: A001
    """Exception for file not found errors."""


@dataclass(frozen=True)
class SRFFile:
    """Immutable data class representing an SRF file."""

    filename: str
    url: str
    file_type: str
    platform: str
    instrument: str
    channel: Optional[str] = None
    description: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        platform: str,
        instrument: str,
        file_type: str,
    ) -> "SRFFile":
        """Create SRFFile from dictionary data."""
        return cls(
            filename=data["filename"],
            url=data["url"],
            file_type=file_type,
            platform=platform,
            instrument=instrument,
            channel=data.get("channel"),
            description=data.get("description"),
        )

    def get_local_path(self, base_dir: Path) -> Path:
        """Get local file path for download."""
        return base_dir / self.platform / self.instrument / self.filename


def _raise_file_not_found_error(default_path: Path, fallback_path: Path) -> None:
    """Raise a FileNotFoundError with helpful message."""
    msg = (
        f"Default SRF index file not found. Tried:\n"
        f"  - {default_path}\n"
        f"  - {fallback_path}\n"
        f"Please provide a YAML file path explicitly."
    )
    raise FileNotFoundError(msg)


def _raise_yaml_validation_error() -> None:
    """Raise a ValueError for empty/invalid YAML."""
    raise YAMLFileError("YAML file is empty or invalid")


def get_default_yaml_path() -> Path:
    """
    Get the default srf_index.yaml path from the package installation.

    Returns
    -------
        Path to the default srf_index.yaml file

    Raises
    ------
        FileNotFoundError: If the default file cannot be found
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    default_path = script_dir / "srf_index.yaml"

    if default_path.exists():
        return default_path

    # Fallback: try the extracted files version
    fallback_path = script_dir / "srf_files_extracted.yaml"
    if fallback_path.exists():
        return fallback_path

    _raise_file_not_found_error(default_path, fallback_path)


def resolve_yaml_path(yaml_file: Optional[Path]) -> Path:
    """
    Resolve the YAML file path, using default if not provided.

    Args:
        yaml_file: Optional path provided by user

    Returns
    -------
        Resolved path to YAML file

    Raises
    ------
        typer.Exit: If file resolution fails
    """
    try:
        if yaml_file is None:
            resolved_path = get_default_yaml_path()
            console.print(f"[dim]Using default SRF index: {resolved_path}[/dim]")
        else:
            if not yaml_file.exists():
                console.print(f"[red]YAML file not found: {yaml_file}[/red]")
                raise typer.Exit(1) from None
            resolved_path = yaml_file

        return resolved_path

    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


def load_srf_data(yaml_path: Path) -> dict[str, Any]:
    """
    Load SRF data from YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns
    -------
        Parsed YAML data

    Raises
    ------
        typer.Exit: If loading fails
    """
    try:
        with yaml_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if data is None:
                _raise_yaml_validation_error()
            return data
    except Exception as e:
        console.print(f"[red]Error loading YAML file {yaml_path}: {e}[/red]")
        raise typer.Exit(1) from e


def extract_srf_files(data: dict[str, Any]) -> list[SRFFile]:
    """
    Extract all SRF files from YAML data using functional approach.

    Args:
        data: Parsed YAML data

    Returns
    -------
        List of SRFFile objects
    """

    def process_file_type(
        file_type: str, platforms: dict[str, Any]
    ) -> Iterator[SRFFile]:
        """Process a single file type and yield SRF files."""
        for platform, instruments in platforms.items():
            for instrument, files in instruments.items():
                for file_data in files:
                    yield SRFFile.from_dict(
                        file_data, platform, instrument, file_type
                    )

    srf_files_data = data.get("srf_files", {})

    # Use functional approach to flatten all file types
    return [
        srf_file
        for file_type, platforms in srf_files_data.items()
        for srf_file in process_file_type(file_type, platforms)
    ]


def filter_by_platform(srf_files: list[SRFFile], platform: str) -> list[SRFFile]:
    """Filter SRF files by platform."""
    return list(filter(lambda f: f.platform.lower() == platform.lower(), srf_files))


def filter_by_instrument(srf_files: list[SRFFile], instrument: str) -> list[SRFFile]:
    """Filter SRF files by instrument."""
    return list(
        filter(lambda f: f.instrument.lower() == instrument.lower(), srf_files)
    )


def filter_by_channel(srf_files: list[SRFFile], channel: str) -> list[SRFFile]:
    """Filter SRF files by channel."""
    return list(filter(lambda f: f.channel == channel, srf_files))


def filter_by_file_type(srf_files: list[SRFFile], file_type: str) -> list[SRFFile]:
    """Filter SRF files by file type."""
    return list(filter(lambda f: f.file_type.lower() == file_type.lower(), srf_files))


FilterFunction = Callable[[list[SRFFile]], list[SRFFile]]


def compose_filters(*filters: FilterFunction) -> FilterFunction:
    """
    Compose multiple filter functions using functional composition.

    Args:
        *filters: Filter functions to compose

    Returns
    -------
        Composed filter function
    """
    return lambda files: reduce(lambda acc, f: f(acc), filters, files)


def build_filter_chain(
    platform: Optional[str] = None,
    instrument: Optional[str] = None,
    channel: Optional[str] = None,
    file_type: Optional[str] = None,
) -> FilterFunction:
    """
    Build a filter chain based on provided criteria using functional composition.

    Args:
        platform: Platform filter
        instrument: Instrument filter
        channel: Channel filter
        file_type: File type filter

    Returns
    -------
        Composed filter function or identity function if no filters
    """
    filters = []

    if platform:
        filters.append(partial(filter_by_platform, platform=platform))
    if instrument:
        filters.append(partial(filter_by_instrument, instrument=instrument))
    if channel:
        filters.append(partial(filter_by_channel, channel=channel))
    if file_type:
        filters.append(partial(filter_by_file_type, file_type=file_type))

    return compose_filters(*filters) if filters else lambda x: x


def get_unique_values(
    srf_files: list[SRFFile], key_func: Callable[[SRFFile], str]
) -> list[str]:
    """
    Get unique values from SRF files using a key function.

    Args:
        srf_files: List of SRF files
        key_func: Function to extract key from SRF file

    Returns
    -------
        Sorted list of unique values
    """
    return sorted(set(filter(None, map(key_func, srf_files))))


def display_available_options(srf_files: list[SRFFile]) -> None:
    """Display available platforms, instruments, and file types."""
    platforms = get_unique_values(srf_files, lambda f: f.platform)
    instruments = get_unique_values(srf_files, lambda f: f.instrument)
    file_types = get_unique_values(srf_files, lambda f: f.file_type)

    table = Table(title="Available Options")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Values", style="green")

    table.add_row("Platforms", ", ".join(platforms))
    table.add_row("Instruments", ", ".join(instruments))
    table.add_row("File Types", ", ".join(file_types))

    console.print(table)


def display_filtered_files(
    srf_files: list[SRFFile], title: str = "Filtered SRF Files"
) -> None:
    """Display filtered SRF files in a table."""
    if not srf_files:
        console.print("[yellow]No files match the specified criteria.[/yellow]")
        return

    table = Table(title=f"{title} ({len(srf_files)} files)")
    table.add_column("Platform", style="cyan")
    table.add_column("Instrument", style="green")
    table.add_column("Filename", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Channel", style="yellow")

    # Sort files for consistent display
    sorted_files = sorted(
        srf_files, key=lambda f: (f.platform, f.instrument, f.filename)
    )

    for srf_file in sorted_files:
        table.add_row(
            srf_file.platform,
            srf_file.instrument,
            srf_file.filename,
            srf_file.file_type,
            srf_file.channel or "N/A",
        )

    console.print(table)


async def download_file(
    session: aiohttp.ClientSession,
    srf_file: SRFFile,
    base_dir: Path,
    progress: Progress,
    task_id: int,
) -> bool:
    """
    Download a single SRF file.

    Args:
        session: HTTP session
        srf_file: SRF file to download
        base_dir: Base directory for downloads
        progress: Rich progress instance
        task_id: Progress task ID

    Returns
    -------
        True if successful, False otherwise
    """
    local_path = srf_file.get_local_path(base_dir)

    # Create directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if file already exists
    if local_path.exists():
        progress.update(task_id, advance=1)
        return True

    try:
        async with session.get(srf_file.url) as response:
            if response.status == HTTP_OK:
                content = await response.read()
                with local_path.open("wb") as f:
                    f.write(content)
                progress.update(task_id, advance=1)
                return True
            else:
                console.print(f"[red]HTTP {response.status} for {srf_file.url}[/red]")
                progress.update(task_id, advance=1)
                return False
    except Exception as e:
        console.print(f"[red]Error downloading {srf_file.filename}: {e}[/red]")
        progress.update(task_id, advance=1)
        return False


async def download_files(
    srf_files: list[SRFFile], output_dir: Path, max_concurrent: int = 5
) -> dict[str, int]:
    """
    Download multiple SRF files concurrently.

    Args:
        srf_files: List of SRF files to download
        output_dir: Output directory
        max_concurrent: Maximum concurrent downloads

    Returns
    -------
        Dictionary with download statistics
    """
    if not srf_files:
        return {"success": 0, "failed": 0, "skipped": 0}

    # Count existing files
    existing_files = sum(
        1 for f in srf_files if f.get_local_path(output_dir).exists()
    )

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Downloading {len(srf_files)} files...",
            total=len(srf_files),
        )

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def download_with_semaphore(srf_file: SRFFile) -> bool:
                async with semaphore:
                    return await download_file(
                        session, srf_file, output_dir, progress, task
                    )

            results = await asyncio.gather(
                *[download_with_semaphore(f) for f in srf_files],
                return_exceptions=True,
            )

    # Calculate statistics
    success_count = sum(1 for r in results if r is True)
    failed_count = sum(1 for r in results if r is False or isinstance(r, Exception))

    return {
        "success": success_count,
        "failed": failed_count,
        "skipped": existing_files,
    }


def load_and_extract_srf_data(yaml_file: Optional[Path]) -> list[SRFFile]:
    """
    Load and extract SRF data from YAML file.

    Args:
        yaml_file: Optional path to YAML file

    Returns
    -------
        List of SRF files
    """
    yaml_path = resolve_yaml_path(yaml_file)
    data = load_srf_data(yaml_path)
    return extract_srf_files(data)


@app.command()
def list_options(
    yaml_file: Optional[Path] = YAML_FILE_ARG,
) -> None:
    """List available platforms, instruments, and file types."""
    srf_files = load_and_extract_srf_data(yaml_file)

    console.print(
        Panel.fit(
            f"[bold green]Total SRF Files: {len(srf_files)}[/bold green]",
            title="SRF Data Summary",
        )
    )

    display_available_options(srf_files)


@app.command()
def search(
    yaml_file: Optional[Path] = YAML_FILE_ARG,
    platform: Optional[str] = PLATFORM_OPT,
    instrument: Optional[str] = INSTRUMENT_OPT,
    channel: Optional[str] = CHANNEL_OPT,
    file_type: Optional[str] = FILE_TYPE_SEARCH_OPT,
) -> None:
    """Search and display SRF files based on criteria."""
    srf_files = load_and_extract_srf_data(yaml_file)

    # Build and apply filter chain
    filter_func = build_filter_chain(platform, instrument, channel, file_type)
    filtered_files = filter_func(srf_files)

    display_filtered_files(filtered_files, "Search Results")


@app.command()
def download(
    yaml_file: Optional[Path] = YAML_FILE_ARG,
    output_dir: Path = OUTPUT_DIR_OPT,
    platform: Optional[str] = PLATFORM_OPT,
    instrument: Optional[str] = INSTRUMENT_OPT,
    channel: Optional[str] = CHANNEL_OPT,
    file_type: Optional[str] = FILE_TYPE_OPT,
    max_concurrent: int = MAX_CONCURRENT_OPT,
    dry_run: bool = DRY_RUN_OPT,
) -> None:
    """Download SRF files based on criteria."""
    srf_files = load_and_extract_srf_data(yaml_file)

    # Build and apply filter chain
    filter_func = build_filter_chain(platform, instrument, channel, file_type)
    filtered_files = filter_func(srf_files)

    if not filtered_files:
        console.print("[yellow]No files match the specified criteria.[/yellow]")
        raise typer.Exit(0)

    # Show what will be downloaded
    display_filtered_files(filtered_files, "Files to Download")

    if dry_run:
        console.print("[blue]Dry run - no files were downloaded.[/blue]")
        raise typer.Exit(0)

    # Confirm download
    if not typer.confirm(f"Download {len(filtered_files)} files to {output_dir}?"):
        raise typer.Exit(0)

    # Download files
    console.print(f"[green]Starting download to {output_dir}...[/green]")

    async def run_download():
        return await download_files(filtered_files, output_dir, max_concurrent)

    stats = asyncio.run(run_download())

    # Display results
    results_table = Table(title="Download Results")
    results_table.add_column("Status", style="cyan")
    results_table.add_column("Count", style="green")

    results_table.add_row("Successful", str(stats["success"]))
    results_table.add_row("Failed", str(stats["failed"]))
    results_table.add_row("Skipped (existing)", str(stats["skipped"]))

    console.print(results_table)

    if stats["failed"] > 0:
        console.print(f"[red]{stats['failed']} downloads failed.[/red]")
        raise typer.Exit(1)
    else:
        console.print("[green]All downloads completed successfully![/green]")


@app.command()
def info(
    yaml_file: Optional[Path] = YAML_FILE_ARG,
) -> None:
    """Show information about the SRF data file."""
    yaml_path = resolve_yaml_path(yaml_file)
    data = load_srf_data(yaml_path)
    srf_files = extract_srf_files(data)

    metadata = data.get("metadata", {})

    # Create info table
    info_table = Table(title="SRF Data Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Source File", str(yaml_path))
    info_table.add_row("Total Files", str(len(srf_files)))
    info_table.add_row("Extraction Date", metadata.get("extraction_date", "Unknown"))
    info_table.add_row("File Types", ", ".join(metadata.get("file_types", [])))
    info_table.add_row("Description", metadata.get("description", "N/A"))

    console.print(info_table)

    # Show file type breakdown using functional approach
    file_type_counts = reduce(
        lambda acc, srf_file: {
            **acc,
            srf_file.file_type: acc.get(srf_file.file_type, 0) + 1,
        },
        srf_files,
        {},
    )

    breakdown_table = Table(title="File Type Breakdown")
    breakdown_table.add_column("File Type", style="magenta")
    breakdown_table.add_column("Count", style="yellow")
    breakdown_table.add_column("Description", style="white")

    descriptions = {
        "txt": "Individual channel files",
        "tar.gz": "Archive files (all channels)",
        "flt": "Filter/passband files",
    }

    for file_type, count in sorted(file_type_counts.items()):
        breakdown_table.add_row(
            file_type,
            str(count),
            descriptions.get(file_type, "Unknown"),
        )

    console.print(breakdown_table)


if __name__ == "__main__":
    app()
