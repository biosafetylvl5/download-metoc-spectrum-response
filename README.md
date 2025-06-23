# SRF File Downloader

A simple, functional Python script to download Spectral Response Function (SRF) files for meteorological and oceanographic satellite instruments using extracted YAML data. Requires python 3.11 or greater.

- Download individual channel files (.txt), archive files (.tar.gz), and filter files (.flt)
- Parallel download with configurable limits

## Installation

```bash
pip install https://github.com/biosafetylvl5/download-metoc-spectrum-response.git
```

## Quick Start

```bash
# Show available
srf-downloader list-options

# Download all AVHRR files
srf-downloader download --platform visible_ir_sensors --instrument avhrr

# Search for specific files
srf-downloader search --platform_type visible_ir_sensors --channel "01"
```

## Commands

### `list-options`
Display available platforms, instruments, and file types.

### `search`
Search and display files matching specified criteria without downloading.

### `download`
Download files matching specified criteria with progress tracking.

### `info`
Show metadata and statistics about the SRF data file.

## Filter Options

- `--platform_type, -p`: Filter by satellite platform type (vis, ir, etc.)
- `--instrument, -i`: Filter by instrument 
- `--channel, -c`: Filter by specific channel
- `--type, -t`: Filter by file type (txt, tar.gz, flt)

## Examples

```bash
# Download specific channel files
srf-downloader download --instrument avhrr --channel "01" --type txt

# Preview downloads without downloading
srf-downloader download --platform visible_ir_sensors --dry-run

# Download with custom output directory and concurrency
srf-downloader download --output ./downloads --concurrent 10
```
Use `poetry run cz commit` for guided commit message creation.
