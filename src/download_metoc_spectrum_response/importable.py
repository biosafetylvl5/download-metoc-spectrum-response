"""
SRF Downloader Convenience Class - Real Catalog Tests.

A high-level Python interface for programmatically downloading and managing
Spectral Response Function (SRF) files from meteorological satellite instruments.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .download_metoc_spectrum_response import (
    SRFFile,
    build_filter_chain,
    download_files,
    extract_srf_files,
    get_unique_values,
    load_srf_data,
    resolve_yaml_path,
)


class SRFDownloader:
    """
    High-level interface for SRF file operations.

    Provides convenient methods for searching, filtering, and downloading
    Spectral Response Function files from meteorological satellite instruments.
    Encapsulates catalog loading and file management operations.

    Parameters
    ----------
    yaml_file : Path, optional
        Path to SRF YAML catalog file. Uses package default if None.
    output_dir : Path, optional
        Default output directory for downloads, by default Path("srf_downloads").
    max_concurrent : int, optional
        Default maximum concurrent downloads, by default 5.

    Attributes
    ----------
    catalog_path : Path
        Resolved path to the SRF catalog file.
    output_dir : Path
        Default output directory for downloads.
    max_concurrent : int
        Default maximum concurrent downloads.
    srf_files : list[SRFFile]
        Loaded SRF files from catalog.

    Examples
    --------
    Basic usage with default catalog:

    >>> downloader = SRFDownloader()
    >>> files = downloader.get_files()
    >>> len(files) > 0
    True
    >>> isinstance(files[0], SRFFile)
    True

    Create with custom settings:

    >>> downloader = SRFDownloader(
    ...     output_dir=Path("/custom/path"),
    ...     max_concurrent=10
    ... )
    >>> downloader.output_dir
    PosixPath('/custom/path')
    >>> downloader.max_concurrent
    10
    """

    def __init__(
        self,
        yaml_file: Path | None = None,
        output_dir: Path = Path("srf_downloads"),
        max_concurrent: int = 5,
    ) -> None:
        """
        Initialize SRF downloader with catalog and settings.

        Parameters
        ----------
        yaml_file : Path, optional
            Path to SRF YAML catalog file.
        output_dir : Path, optional
            Default output directory for downloads.
        max_concurrent : int, optional
            Default maximum concurrent downloads.
        """
        self.catalog_path = resolve_yaml_path(yaml_file)
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self._catalog_data: dict[str, Any] | None = None
        self._srf_files: list[SRFFile] | None = None

    @property
    def catalog_data(self) -> dict[str, Any]:
        """
        Lazy-loaded catalog data from YAML file.

        Returns
        -------
        dict[str, Any]
            Parsed YAML catalog data.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> data = downloader.catalog_data
        >>> "srf_files" in data
        True
        >>> isinstance(data, dict)
        True
        """
        if self._catalog_data is None:
            self._catalog_data = load_srf_data(self.catalog_path)
        return self._catalog_data

    @property
    def srf_files(self) -> list[SRFFile]:
        """
        Lazy-loaded list of all SRF files from catalog.

        Returns
        -------
        list[SRFFile]
            Complete list of SRF files from catalog.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> files = downloader.srf_files
        >>> len(files) > 0
        True
        >>> all(isinstance(f, SRFFile) for f in files)
        True
        >>> all(hasattr(f, 'platform') for f in files)
        True
        >>> all(hasattr(f, 'instrument') for f in files)
        True
        """
        if self._srf_files is None:
            self._srf_files = extract_srf_files(self.catalog_data)
        return self._srf_files

    def get_files(self) -> list[SRFFile]:
        """
        Get all SRF files from catalog.

        Convenience method that returns the complete list of SRF files
        without any filtering applied.

        Returns
        -------
        list[SRFFile]
            Complete list of SRF files.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> files = downloader.get_files()
        >>> isinstance(files, list)
        True
        >>> len(files) > 0
        True
        >>> files == downloader.srf_files
        True
        """
        return self.srf_files

    def search(
        self,
        platform: str | None = None,
        instrument: str | None = None,
        channel: str | None = None,
        file_type: str | None = None,
    ) -> list[SRFFile]:
        """
        Search for SRF files matching specified criteria.

        Applies filters sequentially to find files matching all specified
        criteria. All parameters are optional and case-insensitive where applicable.

        Parameters
        ----------
        platform : str, optional
            Platform name filter (case-insensitive).
        instrument : str, optional
            Instrument name filter (case-insensitive).
        channel : str, optional
            Channel identifier filter (exact match).
        file_type : str, optional
            File type filter (case-insensitive).

        Returns
        -------
        list[SRFFile]
            Filtered list of SRF files matching criteria.

        Examples
        --------
        Search by platform (should find files if platform exists):

        >>> downloader = SRFDownloader()
        >>> platforms = downloader.get_platforms()
        >>> if platforms:
        ...     platform_files = downloader.search(platform=platforms[0])
        ...     len(platform_files) > 0
        ... else:
        ...     True  # No platforms in catalog
        True

        Search by file type:

        >>> downloader = SRFDownloader()
        >>> file_types = downloader.get_file_types()
        >>> if file_types:
        ...     type_files = downloader.search(file_type=file_types[0])
        ...     all(f.file_type.lower() == file_types[0].lower() for f in type_files)
        ... else:
        ...     True  # No file types in catalog
        True

        Search with no criteria returns all files:

        >>> downloader = SRFDownloader()
        >>> all_files = downloader.search()
        >>> all_files == downloader.get_files()
        True
        """
        filter_func = build_filter_chain(platform, instrument, channel, file_type)
        return filter_func(self.srf_files)

    def get_platforms(self) -> list[str]:
        """
        Get list of available platforms in catalog.

        Returns
        -------
        list[str]
            Sorted list of unique platform names.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> platforms = downloader.get_platforms()
        >>> isinstance(platforms, list)
        True
        >>> all(isinstance(p, str) for p in platforms)
        True
        >>> len(platforms) >= 0
        True
        >>> platforms == sorted(set(platforms))  # Should be sorted and unique
        True
        """
        return get_unique_values(self.srf_files, lambda f: f.platform)

    def get_instruments(self, platform: str | None = None) -> list[str]:
        """
        Get list of available instruments, optionally filtered by platform.

        Parameters
        ----------
        platform : str, optional
            Platform name to filter instruments. If None, returns all instruments.

        Returns
        -------
        list[str]
            Sorted list of unique instrument names.

        Examples
        --------
        Get all instruments:

        >>> downloader = SRFDownloader()
        >>> instruments = downloader.get_instruments()
        >>> isinstance(instruments, list)
        True
        >>> all(isinstance(i, str) for i in instruments)
        True

        Get instruments for specific platform:

        >>> downloader = SRFDownloader()
        >>> platforms = downloader.get_platforms()
        >>> if platforms:
        ...     platform_instruments = downloader.get_instruments(platforms[0])
        ...     isinstance(platform_instruments, list)
        ... else:
        ...     True  # No platforms available
        True

        Filtered instruments should be subset of all instruments:

        >>> downloader = SRFDownloader()
        >>> all_instruments = set(downloader.get_instruments())
        >>> platforms = downloader.get_platforms()
        >>> if platforms:
        ...     platform_instruments = set(downloader.get_instruments(platforms[0]))
        ...     platform_instruments.issubset(all_instruments)
        ... else:
        ...     True  # No platforms to test
        True
        """
        files = self.search(platform=platform) if platform else self.srf_files
        return get_unique_values(files, lambda f: f.instrument)

    def get_channels(
        self,
        platform: str | None = None,
        instrument: str | None = None,
    ) -> list[str]:
        """
        Get list of available channels, optionally filtered by platform and instrument.

        Parameters
        ----------
        platform : str, optional
            Platform name filter.
        instrument : str, optional
            Instrument name filter.

        Returns
        -------
        list[str]
            Sorted list of unique channel identifiers.

        Examples
        --------
        Get all channels:

        >>> downloader = SRFDownloader()
        >>> channels = downloader.get_channels()
        >>> isinstance(channels, list)
        True
        >>> all(isinstance(c, str) for c in channels)
        True

        Get channels for specific instrument:

        >>> downloader = SRFDownloader()
        >>> instruments = downloader.get_instruments()
        >>> if instruments:
        ...     inst_channels = downloader.get_channels(instrument=instruments[0])
        ...     isinstance(inst_channels, list)
        ... else:
        ...     True  # No instruments available
        True
        """
        files = self.search(platform=platform, instrument=instrument)
        return get_unique_values(files, lambda f: f.channel or "")

    def get_file_types(self) -> list[str]:
        """
        Get list of available file types in catalog.

        Returns
        -------
        list[str]
            Sorted list of unique file types.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> file_types = downloader.get_file_types()
        >>> isinstance(file_types, list)
        True
        >>> len(file_types) > 0
        True
        >>> all(isinstance(ft, str) for ft in file_types)
        True
        >>> file_types == sorted(set(file_types))  # Should be sorted and unique
        True
        """
        return get_unique_values(self.srf_files, lambda f: f.file_type)

    def count_files(
        self,
        platform: str | None = None,
        instrument: str | None = None,
        channel: str | None = None,
        file_type: str | None = None,
    ) -> int:
        """
        Count files matching specified criteria.

        Useful for checking how many files would be downloaded before
        actually performing the download operation.

        Parameters
        ----------
        platform : str, optional
            Platform name filter.
        instrument : str, optional
            Instrument name filter.
        channel : str, optional
            Channel identifier filter.
        file_type : str, optional
            File type filter.

        Returns
        -------
        int
            Number of files matching criteria.

        Examples
        --------
        Count all files:

        >>> downloader = SRFDownloader()
        >>> total = downloader.count_files()
        >>> total == len(downloader.get_files())
        True
        >>> total >= 0
        True

        Count files for specific criteria:

        >>> downloader = SRFDownloader()
        >>> platforms = downloader.get_platforms()
        >>> if platforms:
        ...     platform_count = downloader.count_files(platform=platforms[0])
        ...     platform_count >= 0
        ... else:
        ...     True  # No platforms to test
        True

        Filtered count should be <= total count:

        >>> downloader = SRFDownloader()
        >>> total = downloader.count_files()
        >>> file_types = downloader.get_file_types()
        >>> if file_types:
        ...     type_count = downloader.count_files(file_type=file_types[0])
        ...     type_count <= total
        ... else:
        ...     True  # No file types to test
        True
        """
        return len(self.search(platform, instrument, channel, file_type))

    async def download(
        self,
        platform: str | None = None,
        instrument: str | None = None,
        channel: str | None = None,
        file_type: str | None = None,
        output_dir: Path | None = None,
        max_concurrent: int | None = None,
    ) -> dict[str, int]:
        """
        Download SRF files matching criteria asynchronously.

        Filters files based on criteria and downloads them with progress tracking.
        Creates hierarchical directory structure for organized storage.

        Parameters
        ----------
        platform : str, optional
            Platform name filter.
        instrument : str, optional
            Instrument name filter.
        channel : str, optional
            Channel identifier filter.
        file_type : str, optional
            File type filter.
        output_dir : Path, optional
            Output directory. Uses instance default if None.
        max_concurrent : int, optional
            Maximum concurrent downloads. Uses instance default if None.

        Returns
        -------
        dict[str, int]
            Download statistics with keys 'success', 'failed', 'skipped'.

        Examples
        --------
        Test with no matching files:

        >>> import asyncio
        >>> async def test_empty_download():
        ...     downloader = SRFDownloader()
        ...     # Search for non-existent platform
        ...     stats = await downloader.download(platform="NONEXISTENT")
        ...     return stats
        >>> stats = asyncio.run(test_empty_download())
        >>> stats == {"success": 0, "failed": 0, "skipped": 0}
        True
        """
        files_to_download = self.search(platform, instrument, channel, file_type)

        if not files_to_download:
            return {"success": 0, "failed": 0, "skipped": 0}

        target_dir = output_dir or self.output_dir
        concurrent_limit = max_concurrent or self.max_concurrent

        return await download_files(files_to_download, target_dir, concurrent_limit)

    def download_sync(
        self,
        platform: str | None = None,
        instrument: str | None = None,
        channel: str | None = None,
        file_type: str | None = None,
        output_dir: Path | None = None,
        max_concurrent: int | None = None,
    ) -> dict[str, int]:
        """
        Download SRF files matching criteria synchronously.

        Synchronous wrapper around the async download method for convenience
        when working in non-async contexts.

        Parameters
        ----------
        platform : str, optional
            Platform name filter.
        instrument : str, optional
            Instrument name filter.
        channel : str, optional
            Channel identifier filter.
        file_type : str, optional
            File type filter.
        output_dir : Path, optional
            Output directory. Uses instance default if None.
        max_concurrent : int, optional
            Maximum concurrent downloads. Uses instance default if None.

        Returns
        -------
        dict[str, int]
            Download statistics with keys 'success', 'failed', 'skipped'.

        Examples
        --------
        Test synchronous download with no matches:

        >>> downloader = SRFDownloader()
        >>> stats = downloader.download_sync(platform="NONEXISTENT")
        >>> stats == {"success": 0, "failed": 0, "skipped": 0}
        True
        """
        return asyncio.run(
            self.download(
                platform,
                instrument,
                channel,
                file_type,
                output_dir,
                max_concurrent,
            ),
        )

    def get_catalog_info(self) -> dict[str, Any]:
        """
        Get metadata information about the SRF catalog.

        Returns catalog metadata including extraction date, file types,
        and descriptive information when available.

        Returns
        -------
        dict[str, Any]
            Catalog metadata dictionary.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> info = downloader.get_catalog_info()
        >>> isinstance(info, dict)
        True
        """
        return self.catalog_data.get("metadata", {})

    def get_file_by_filename(self, filename: str) -> SRFFile | None:
        """
        Find SRF file by exact filename match.

        Searches through all files in catalog for exact filename match.
        Returns first match found.

        Parameters
        ----------
        filename : str
            Exact filename to search for.

        Returns
        -------
        SRFFile or None
            Matching SRF file or None if not found.

        Examples
        --------
        Test with known non-existent filename:

        >>> downloader = SRFDownloader()
        >>> file = downloader.get_file_by_filename("nonexistent_file.txt")
        >>> file is None
        True

        Test that method returns correct type when file exists:

        >>> downloader = SRFDownloader()
        >>> files = downloader.get_files()
        >>> if files:
        ...     found_file = downloader.get_file_by_filename(files[0].filename)
        ...     found_file == files[0]
        ... else:
        ...     True  # No files to test
        True
        """
        for srf_file in self.srf_files:
            if srf_file.filename == filename:
                return srf_file
        return None

    def get_files_by_pattern(self, pattern: str) -> list[SRFFile]:
        """
        Find SRF files with filenames matching pattern.

        Uses simple string containment matching. For more complex pattern
        matching, use search() method with appropriate filters.

        Parameters
        ----------
        pattern : str
            Pattern to search for in filenames (case-sensitive).

        Returns
        -------
        list[SRFFile]
            List of SRF files with filenames containing pattern.

        Examples
        --------
        Test with pattern that shouldn't match anything:

        >>> downloader = SRFDownloader()
        >>> no_match = downloader.get_files_by_pattern("XYZNOMATCH")
        >>> len(no_match)
        0

        Test that results contain the pattern:

        >>> downloader = SRFDownloader()
        >>> files = downloader.get_files()
        >>> if files:
        ...     # Use first character of first filename as pattern
        ...     pattern = files[0].filename[0]
        ...     matches = downloader.get_files_by_pattern(pattern)
        ...     all(pattern in f.filename for f in matches)
        ... else:
        ...     True  # No files to test
        True
        """
        return [f for f in self.srf_files if pattern in f.filename]

    def reload_catalog(self, yaml_file: Path | None = None) -> None:
        """
        Reload catalog data from file.

        Clears cached data and reloads from specified file or current catalog path.
        Useful when catalog file has been updated externally.

        Parameters
        ----------
        yaml_file : Path, optional
            New catalog file path. If None, reloads current catalog.

        Examples
        --------
        Test reload functionality:

        >>> downloader = SRFDownloader()
        >>> original_count = len(downloader)
        >>> downloader.reload_catalog()
        >>> len(downloader) == original_count
        True
        """
        if yaml_file is not None:
            self.catalog_path = resolve_yaml_path(yaml_file)

        # Clear cached data to force reload
        self._catalog_data = None
        self._srf_files = None

    def __len__(self) -> int:
        """
        Return total number of SRF files in catalog.

        Returns
        -------
        int
            Total number of SRF files.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> total_files = len(downloader)
        >>> total_files >= 0
        True
        >>> total_files == len(downloader.get_files())
        True
        """
        return len(self.srf_files)

    def __repr__(self) -> str:
        """
        Return string representation of SRFDownloader.

        Returns
        -------
        str
            String representation showing catalog path and file count.

        Examples
        --------
        >>> downloader = SRFDownloader()
        >>> repr_str = repr(downloader)
        >>> "SRFDownloader" in repr_str
        True
        >>> "files=" in repr_str
        True
        >>> "srf_catalog.yaml" in repr_str
        True
        """
        return (
            f"SRFDownloader(catalog='{self.catalog_path.name}', "
            f"files={len(self)}, output_dir='{self.output_dir}')"
        )
