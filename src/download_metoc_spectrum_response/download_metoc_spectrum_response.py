#!/usr/bin/env python3
"""
SRF File Downloader.

A command-line tool for downloading Spectral Response Function (SRF) files
from meteorological satellite instruments based on YAML catalog data.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Annotated, Any, Union

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

# Defining these because typer does not support the new | syntax for optionals/unions
PathOrNone = Union[Path, None]  # noqa: UP007
StrOrNone = Union[str, None]  # noqa: UP007


class SRFFileError(Exception):
    """Base exception for SRF file operations."""


class YAMLFileError(SRFFileError):
    """Exception raised when YAML file operations fail."""


class FileNotFoundError(SRFFileError):  # noqa: A001
    """Exception raised when required files cannot be located."""


@dataclass(frozen=True)
class SRFFile:
    """
    Immutable representation of a Spectral Response Function file.

    This class encapsulates metadata for SRF files including download URLs,
    platform information, and file characteristics.

    Parameters
    ----------
    filename : str
        Name of the SRF file including extension.
    url : str
        Complete HTTP/HTTPS URL for downloading the file.
    file_type : str
        Type of file (e.g., 'txt', 'tar.gz', 'flt').
    platform : str
        Satellite platform name (e.g., 'NOAA-20', 'Sentinel-3A').
    instrument : str
        Instrument name (e.g., 'VIIRS', 'OLCI').
    channel : str, optional
        Specific channel identifier, by default None.
    description : str, optional
        Human-readable description of the file, by default None.

    Examples
    --------
    >>> srf = SRFFile(
    ...     filename="viirs_npp_ch1.txt",
    ...     url="https://example.com/viirs_npp_ch1.txt",
    ...     file_type="txt",
    ...     platform="NPP",
    ...     instrument="VIIRS",
    ...     channel="M01"
    ... )
    >>> srf.filename
    'viirs_npp_ch1.txt'
    >>> srf.platform
    'NPP'
    """

    filename: str
    url: str
    file_type: str
    platform: str
    instrument: str
    channel: StrOrNone = None
    description: StrOrNone = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        platform: str,
        instrument: str,
        file_type: str,
    ) -> SRFFile:
        """
        Create SRFFile instance from dictionary data.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing file metadata with required keys 'filename'
            and 'url'. Optional keys include 'channel' and 'description'.
        platform : str
            Satellite platform identifier.
        instrument : str
            Instrument identifier.
        file_type : str
            File type classification.

        Returns
        -------
        SRFFile
            New SRFFile instance with populated metadata.

        Examples
        --------
        >>> data = {
        ...     'filename': 'test.txt',
        ...     'url': 'https://example.com/test.txt',
        ...     'channel': 'M01'
        ... }
        >>> srf = SRFFile.from_dict(data, 'NPP', 'VIIRS', 'txt')
        >>> srf.channel
        'M01'
        """
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
        """
        Generate local filesystem path for downloaded file.

        Creates a hierarchical directory structure: base_dir/platform/instrument/filename

        Parameters
        ----------
        base_dir : Path
            Root directory for file storage.

        Returns
        -------
        Path
            Complete local path where file should be stored.

        Examples
        --------
        >>> srf = SRFFile('test.txt', 'http://example.com', 'txt', 'NPP', 'VIIRS')
        >>> path = srf.get_local_path(Path('/downloads'))
        >>> str(path)
        '/downloads/NPP/VIIRS/test.txt'
        """
        return base_dir / self.platform / self.instrument / self.filename


def _raise_file_not_found_error(default_path: Path) -> None:
    """
    Raise FileNotFoundError with detailed path information.

    Parameters
    ----------
    default_path : Path
        Primary path that was attempted.

    Raises
    ------
    FileNotFoundError
        Always raised with descriptive message listing attempted paths.
    """
    msg = (
        f"Default SRF index file not found. Tried:"
        f" '{default_path}' "
        f"Please provide a YAML file path explicitly."
    )
    raise FileNotFoundError(msg)


def _raise_yaml_validation_error() -> None:
    """
    Raise YAMLFileError for invalid YAML content.

    Raises
    ------
    YAMLFileError
        Always raised when YAML file is empty or contains invalid data.
    """
    raise YAMLFileError("YAML file is empty or invalid")


def get_default_yaml_path() -> Path:
    """
    Locate default SRF index YAML file in package installation.

    Searches for 'srf_catalog.yaml' in the script's directory.

    Returns
    -------
    Path
        Path to the located YAML file.

    Raises
    ------
    FileNotFoundError
        When neither default file can be found.

    Examples
    --------
    >>> # Assuming srf_catalog.yaml exists in script directory
    >>> path = get_default_yaml_path()  # doctest: +SKIP
    >>> path.name  # doctest: +SKIP
    'srf_catalog.yaml'
    """
    script_dir = Path(__file__).parent
    default_path = script_dir / "srf_catalog.yaml"

    if default_path.exists():
        return default_path

    _raise_file_not_found_error(default_path)


def resolve_yaml_path(yaml_file: PathOrNone) -> Path:
    """
    Resolve YAML file path, using default when none provided.

    Parameters
    ----------
    yaml_file : Path or None
        User-provided path to YAML file. If None, searches for default.

    Returns
    -------
    Path
        Resolved path to existing YAML file.

    Raises
    ------
    typer.Exit
        When file resolution fails or specified file doesn't exist.

    Examples
    --------
    >>> # Test with None (uses default)
    >>> path = resolve_yaml_path(None)  # doctest: +SKIP

    >>> # Test with existing file
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
    ...     path = resolve_yaml_path(Path(f.name))
    ...     isinstance(path, Path)
    True
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
    Load and parse SRF data from YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML file containing SRF catalog data.

    Returns
    -------
    dict[str, Any]
        Parsed YAML data structure.

    Raises
    ------
    typer.Exit
        When file loading or parsing fails.

    Examples
    --------
    >>> import tempfile
    >>> import yaml as yaml_module
    >>> test_data = {'srf_files': {'txt': {}}}
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    ...     yaml_module.dump(test_data, f)
    ...     temp_path = Path(f.name)
    >>> data = load_srf_data(temp_path)
    >>> 'srf_files' in data
    True
    >>> temp_path.unlink()  # cleanup
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
    Extract SRF file objects from parsed YAML data.

    Processes nested YAML structure to create flat list of SRFFile objects.
    Expected structure: data['srf_files'][file_type][platform][instrument][files].

    Parameters
    ----------
    data : dict[str, Any]
        Parsed YAML data containing SRF file catalog.

    Returns
    -------
    list[SRFFile]
        Flat list of all SRF files found in the catalog.

    Examples
    --------
    >>> test_data = {
    ...     'srf_files': {
    ...         'txt': {
    ...             'NPP': {
    ...                 'VIIRS': [
    ...                     {'filename': 'test.txt', 'url': 'http://example.com/test.txt'}
    ...                 ]
    ...             }
    ...         }
    ...     }
    ... }
    >>> files = extract_srf_files(test_data)
    >>> len(files)
    1
    >>> files[0].platform
    'NPP'
    """

    def process_file_type(
        file_type: str,
        platforms: dict[str, Any],
    ) -> Iterator[SRFFile]:
        """
        Process single file type and yield SRF files.

        Parameters
        ----------
        file_type : str
            Type of files being processed.
        platforms : dict[str, Any]
            Platform type data containing instruments and files.

        Yields
        ------
        SRFFile
            Individual SRF file objects.
        """
        for platform, instruments in platforms.items():
            for instrument, files in instruments.items():
                for file_data in files:
                    yield SRFFile.from_dict(
                        file_data,
                        platform,
                        instrument,
                        file_type,
                    )

    srf_files_data = data.get("srf_files", {})

    return [
        srf_file
        for file_type, platforms in srf_files_data.items()
        for srf_file in process_file_type(file_type, platforms)
    ]


def filter_by_platform(srf_files: list[SRFFile], platform: str) -> list[SRFFile]:
    """
    Filter SRF files by platform name.

    Performs case-insensitive matching against platform names.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to filter.
    platform : str
        Platform type name to match (case-insensitive).

    Returns
    -------
    list[SRFFile]
        Filtered list containing only files from specified platform.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NOAA-20', 'VIIRS')
    ... ]
    >>> filtered = filter_by_platform(files, 'npp')
    >>> len(filtered)
    1
    >>> filtered[0].platform
    'NPP'
    """
    return list(filter(lambda f: f.platform.lower() == platform.lower(), srf_files))


def filter_by_instrument(srf_files: list[SRFFile], instrument: str) -> list[SRFFile]:
    """
    Filter SRF files by instrument name.

    Performs case-insensitive matching against instrument names.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to filter.
    instrument : str
        Instrument name to match (case-insensitive).

    Returns
    -------
    list[SRFFile]
        Filtered list containing only files from specified instrument.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NPP', 'ATMS')
    ... ]
    >>> filtered = filter_by_instrument(files, 'VIIRS')
    >>> len(filtered)
    1
    >>> filtered[0].instrument
    'VIIRS'
    """
    return list(
        filter(lambda f: f.instrument.lower() == instrument.lower(), srf_files),
    )


def filter_by_channel(srf_files: list[SRFFile], channel: str) -> list[SRFFile]:
    """
    Filter SRF files by channel identifier.

    Performs exact string matching against channel identifiers.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to filter.
    channel : str
        Channel identifier to match exactly.

    Returns
    -------
    list[SRFFile]
        Filtered list containing only files from specified channel.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS', 'M01'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NPP', 'VIIRS', 'M02')
    ... ]
    >>> filtered = filter_by_channel(files, 'M01')
    >>> len(filtered)
    1
    >>> filtered[0].channel
    'M01'
    """
    return list(filter(lambda f: f.channel == channel, srf_files))


def filter_by_file_type(srf_files: list[SRFFile], file_type: str) -> list[SRFFile]:
    """
    Filter SRF files by file type.

    Performs case-insensitive matching against file types.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to filter.
    file_type : str
        File type to match (case-insensitive).

    Returns
    -------
    list[SRFFile]
        Filtered list containing only files of specified type.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS'),
    ...     SRFFile('f2.tar.gz', 'url2', 'tar.gz', 'NPP', 'VIIRS')
    ... ]
    >>> filtered = filter_by_file_type(files, 'TXT')
    >>> len(filtered)
    1
    >>> filtered[0].file_type
    'txt'
    """
    return list(filter(lambda f: f.file_type.lower() == file_type.lower(), srf_files))


FilterFunction = Callable[[list[SRFFile]], list[SRFFile]]


def compose_filters(*filters: FilterFunction) -> FilterFunction:
    """
    Compose multiple filter functions into single function.

    Creates a pipeline where each filter is applied sequentially to the
    output of the previous filter.

    Parameters
    ----------
    *filters : FilterFunction
        Variable number of filter functions to compose.

    Returns
    -------
    FilterFunction
        Composed filter function that applies all filters in sequence.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS', 'M01'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NOAA-20', 'VIIRS', 'M02')
    ... ]
    >>> platform_filter = lambda fs: filter_by_platform(fs, 'NPP')
    >>> channel_filter = lambda fs: filter_by_channel(fs, 'M01')
    >>> composed = compose_filters(platform_filter, channel_filter)
    >>> result = composed(files)
    >>> len(result)
    1
    """
    return lambda files: reduce(lambda acc, f: f(acc), filters, files)


def build_filter_chain(
    platform: StrOrNone = None,
    instrument: StrOrNone = None,
    channel: StrOrNone = None,
    file_type: StrOrNone = None,
) -> FilterFunction:
    """
    Build filter chain based on provided criteria.

    Creates a composed filter function that applies all non-None criteria
    in sequence. Returns identity function when no filters specified.

    Parameters
    ----------
    platform : str, optional
        Platform type filter, by default None.
    instrument : str, optional
        Instrument name filter, by default None.
    channel : str, optional
        Channel identifier filter, by default None.
    file_type : str, optional
        File type filter, by default None.

    Returns
    -------
    FilterFunction
        Composed filter function or identity function if no filters.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS', 'M01'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NOAA-20', 'VIIRS', 'M02')
    ... ]
    >>> filter_func = build_filter_chain(platform='NPP', channel='M01')
    >>> result = filter_func(files)
    >>> len(result)
    1
    >>> result[0].platform
    'NPP'
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
    srf_files: list[SRFFile],
    key_func: Callable[[SRFFile], str],
) -> list[str]:
    """
    Extract unique values from SRF files using key function.

    Applies key function to each file, filters out None values,
    and returns sorted list of unique results.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to process.
    key_func : Callable[[SRFFile], str]
        Function to extract value from each SRF file.

    Returns
    -------
    list[str]
        Sorted list of unique non-None values.

    Examples
    --------
    >>> files = [
    ...     SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS'),
    ...     SRFFile('f2.txt', 'url2', 'txt', 'NOAA-20', 'VIIRS'),
    ...     SRFFile('f3.txt', 'url3', 'txt', 'NPP', 'ATMS')
    ... ]
    >>> platforms = get_unique_values(files, lambda f: f.platform)
    >>> platforms
    ['NOAA-20', 'NPP']
    """
    return sorted(set(filter(None, map(key_func, srf_files))))


def display_available_options(srf_files: list[SRFFile]) -> None:
    """
    Display available platforms, instruments, and file types in table.

    Extracts unique values for each category and presents them in a
    formatted Rich table for user reference.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to analyze for available options.

    Examples
    --------
    >>> files = [SRFFile('f1.txt', 'url1', 'txt', 'NPP', 'VIIRS')]
    >>> display_available_options(files)  # doctest: +SKIP
    """
    platforms = get_unique_values(srf_files, lambda f: f.platform)
    instruments = get_unique_values(srf_files, lambda f: f.instrument)
    file_types = get_unique_values(srf_files, lambda f: f.file_type)

    table = Table(title="Available Options")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Values", style="green")

    table.add_row("Platform Types", ", ".join(platforms))
    table.add_row("Instruments", ", ".join(instruments))
    table.add_row("File Types", ", ".join(file_types))

    console.print(table)


def display_filtered_files(
    srf_files: list[SRFFile],
    title: str = "Filtered SRF Files",
) -> None:
    """
    Display SRF files in formatted table.

    Shows file details including platform, instrument, filename, type,
    and channel in a Rich table. Displays message when no files match.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to display.
    title : str, optional
        Table title, by default "Filtered SRF Files".

    Examples
    --------
    >>> files = [SRFFile('test.txt', 'url', 'txt', 'NPP', 'VIIRS', 'M01')]
    >>> display_filtered_files(files, "Test Files")  # doctest: +SKIP
    """
    if not srf_files:
        console.print("[yellow]No files match the specified criteria.[/yellow]")
        return

    table = Table(title=f"{title} ({len(srf_files)} files)")
    table.add_column("Platform Types", style="cyan")
    table.add_column("Instrument", style="green")
    table.add_column("Filename", style="blue")
    table.add_column("Type", style="magenta")
    table.add_column("Channel", style="yellow")

    # Sort files for consistent display
    sorted_files = sorted(
        srf_files,
        key=lambda f: (f.platform, f.instrument, f.filename),
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
    Download single SRF file asynchronously.

    Downloads file from URL to local path, creating directories as needed.
    Skips download if file already exists locally.

    Parameters
    ----------
    session : aiohttp.ClientSession
        HTTP session for making requests.
    srf_file : SRFFile
        SRF file metadata including download URL.
    base_dir : Path
        Base directory for file storage.
    progress : Progress
        Rich progress tracker for UI updates.
    task_id : int
        Progress task identifier for updates.

    Returns
    -------
    bool
        True if download succeeded or file exists, False if failed.

    Examples
    --------
    >>> import asyncio
    >>> async def test_download():
    ...     # This would require actual HTTP session and progress objects
    ...     pass  # doctest: +SKIP
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
    srf_files: list[SRFFile],
    output_dir: Path,
    max_concurrent: int = 5,
) -> dict[str, int]:
    """
    Download multiple SRF files concurrently.

    Manages concurrent downloads with connection limits and progress tracking.
    Counts existing files and provides detailed statistics.

    Parameters
    ----------
    srf_files : list[SRFFile]
        List of SRF files to download.
    output_dir : Path
        Output directory for downloaded files.
    max_concurrent : int, optional
        Maximum concurrent downloads, by default 5.

    Returns
    -------
    dict[str, int]
        Download statistics with keys 'success', 'failed', 'skipped'.

    Examples
    --------
    >>> import asyncio
    >>> async def test_downloads():
    ...     files = []  # Empty list for test
    ...     stats = await download_files(files, Path('/tmp'))
    ...     return stats['success'] == 0
    >>> asyncio.run(test_downloads())
    True
    """
    if not srf_files:
        return {"success": 0, "failed": 0, "skipped": 0}

    # Count existing files
    existing_files = sum(1 for f in srf_files if f.get_local_path(output_dir).exists())

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
            connector=connector,
            timeout=timeout,
        ) as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def download_with_semaphore(srf_file: SRFFile) -> bool:
                async with semaphore:
                    return await download_file(
                        session,
                        srf_file,
                        output_dir,
                        progress,
                        task,
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


def load_and_extract_srf_data(yaml_file: PathOrNone) -> list[SRFFile]:
    """
    Load YAML file and extract SRF file objects.

    Convenience function that combines YAML loading and SRF extraction
    into single operation.

    Parameters
    ----------
    yaml_file : Path or None
        Path to YAML file, or None to use default.

    Returns
    -------
    list[SRFFile]
        List of extracted SRF file objects.

    Examples
    --------
    >>> # This would require actual YAML file
    >>> files = load_and_extract_srf_data(None)  # doctest: +SKIP
    >>> isinstance(files, list)  # doctest: +SKIP
    True
    """
    yaml_path = resolve_yaml_path(yaml_file)
    data = load_srf_data(yaml_path)
    return extract_srf_files(data)


@app.command()
def list_options(
    yaml_file: Annotated[
        PathOrNone,
        typer.Option(
            "--srf-yaml-catalog",
            "-y",
            help="Path to SRF YAML file (uses default if not provided)",
        ),
    ] = None,
) -> None:
    """
    List available platforms, instruments, and file types.

    Displays summary of total files and categorized options available
    in the SRF catalog for filtering and selection.

    Parameters
    ----------
    yaml_file : Path, optional
        Path to SRF YAML catalog file. Uses package default if not specified.

    Examples
    --------
    Command line usage:
        $ srf-downloader list-options
        $ srf-downloader list-options -y custom_catalog.yaml
    """
    srf_files = load_and_extract_srf_data(yaml_file)

    console.print(
        Panel.fit(
            f"[bold green]Total SRF Files: {len(srf_files)}[/bold green]",
            title="SRF Data Summary",
        ),
    )

    display_available_options(srf_files)


@app.command()
def search(
    yaml_file: Annotated[
        PathOrNone,
        typer.Option(
            "--srf-yaml-catalog",
            "-y",
            help="Path to SRF YAML file (uses default if not provided)",
        ),
    ] = None,
    platform: Annotated[
        StrOrNone,
        typer.Option("--platform", "-p", help="Filter by platform"),
    ] = None,
    instrument: Annotated[
        StrOrNone,
        typer.Option("--instrument", "-i", help="Filter by instrument"),
    ] = None,
    channel: Annotated[
        StrOrNone,
        typer.Option("--channel", "-c", help="Filter by channel"),
    ] = None,
    file_type: Annotated[
        StrOrNone,
        typer.Option("--type", "-t", help="Filter by file type (txt, tar.gz, flt)"),
    ] = None,
) -> None:
    """
    Search and display SRF files matching specified criteria.

    Applies filters in sequence and displays results in formatted table.
    All filters are optional and case-insensitive where applicable.

    Parameters
    ----------
    yaml_file : Path, optional
        Path to SRF YAML catalog file.
    platform : str, optional
        Platform type name filter (case-insensitive).
    instrument : str, optional
        Instrument name filter (case-insensitive).
    channel : str, optional
        Channel identifier filter (exact match).
    file_type : str, optional
        File type filter (case-insensitive).

    Examples
    --------
    Command line usage:
        $ srf-downloader search --platform NPP
        $ srf-downloader search -p NOAA-20 -i VIIRS -t txt
        $ srf-downloader search --channel M01
    """
    srf_files = load_and_extract_srf_data(yaml_file)

    # Build and apply filter chain
    filter_func = build_filter_chain(platform, instrument, channel, file_type)
    filtered_files = filter_func(srf_files)

    display_filtered_files(filtered_files, "Search Results")


@app.command()
def download(
    yaml_file: Annotated[
        PathOrNone,
        typer.Option(
            "--srf-yaml-catalog",
            "-y",
            help="Path to SRF YAML file (uses default if not provided)",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path(DEFAULT_OUTPUT_DIR),
    platform: Annotated[
        StrOrNone,
        typer.Option("--platform", "-p", help="Filter by platform"),
    ] = None,
    instrument: Annotated[
        StrOrNone,
        typer.Option("--instrument", "-i", help="Filter by instrument"),
    ] = None,
    channel: Annotated[
        StrOrNone,
        typer.Option("--channel", "-c", help="Filter by channel"),
    ] = None,
    file_type: Annotated[
        StrOrNone,
        typer.Option("--file-type", "-f", help="Filter by file type"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--concurrent", help="Maximum concurrent downloads"),
    ] = DEFAULT_MAX_CONCURRENT,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be downloaded without downloading",
        ),
    ] = False,
) -> None:
    """
    Download SRF files matching specified criteria.

    Filters files, displays selection, and downloads with progress tracking.
    Creates hierarchical directory structure: output_dir/platform/instrument/.

    Parameters
    ----------
    yaml_file : Path, optional
        Path to SRF YAML catalog file.
    output_dir : Path, optional
        Output directory for downloads, by default 'srf_downloads'.
    platform : str, optional
        Platform type name filter.
    instrument : str, optional
        Instrument name filter.
    channel : str, optional
        Channel identifier filter.
    file_type : str, optional
        File type filter.
    max_concurrent : int, optional
        Maximum concurrent downloads, by default 5.
    dry_run : bool, optional
        Show files without downloading, by default False.

    Examples
    --------
    Command line usage:
        $ srf-downloader download --platform NPP
        $ srf-downloader download -p NOAA-20 -o /data/srf --concurrent 10
        $ srf-downloader download --dry-run -i VIIRS
    """
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
    yaml_file: Annotated[
        PathOrNone,
        typer.Option(
            "--srf-yaml-catalog",
            "-y",
            help="Path to SRF YAML file (uses default if not provided)",
        ),
    ] = None,
) -> None:
    """
    Display detailed information about SRF catalog file.

    Shows catalog metadata, file counts, and breakdown by file type
    with descriptions of each type's purpose.

    Parameters
    ----------
    yaml_file : Path, optional
        Path to SRF YAML catalog file.

    Examples
    --------
    Command line usage:
        $ srf-downloader info
        $ srf-downloader info -y custom_catalog.yaml
    """
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
