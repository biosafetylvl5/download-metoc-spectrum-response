# SRF File Downloader

A simple, functional Python script to download Spectral Response Function (SRF) files from meteorological and oceanographic satellite instruments using extracted YAML data.

## Features

- **Multi-format support**: Download individual channel files (.txt), archive files (.tar.gz), and filter files (.flt)
- **Flexible filtering**: Filter by platform, instrument, channel, or file type
- **Concurrent downloads**: Fast parallel downloading with configurable limits
- **Rich CLI interface**: Beautiful progress bars, tables, and colored output
- **Functional design**: Clean, composable filter functions following functional programming principles

## Installation

```bash
# Install with Poetry (recommended)
poetry install

# Or install with pip
pip install aiohttp PyYAML typer rich
```

## Quick Start

```bash
# Show available options
srf-downloader list-options srf_data.yaml

# Download all AVHRR files
srf-downloader download srf_data.yaml --platform visible_ir_sensors --instrument avhrr

# Search for specific files
srf-downloader search srf_data.yaml --platform visible_ir_sensors --channel "01"
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

- `--platform, -p`: Filter by satellite platform
- `--instrument, -i`: Filter by instrument type
- `--channel, -c`: Filter by specific channel
- `--type, -t`: Filter by file type (txt, tar.gz, flt)

## Examples

```bash
# Download specific channel files
srf-downloader download srf_data.yaml --instrument avhrr --channel "01" --type txt

# Preview downloads without downloading
srf-downloader download srf_data.yaml --platform visible_ir_sensors --dry-run

# Download with custom output directory and concurrency
srf-downloader download srf_data.yaml --output ./downloads --concurrent 10
```

## Requirements

- Python ≥3.8
- aiohttp ≥3.8.0
- PyYAML ≥6.0
- typer ≥0.9.0
- rich ≥13.0.0

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and ensure tests pass
4. Use conventional commits (`feat:`, `fix:`, etc.)
5. Submit a pull request

### Development Setup

**Using Dev Containers (Recommended):**
1. Install Docker Desktop and VS Code Dev Containers extension
2. Clone repository and open in VS Code
3. Select "Reopen in Container" when prompted

**Manual Setup:**
```bash
git clone https://github.com/biosafetylvl5/download-metoc-spectrum-response.git
cd download-metoc-spectrum-response
poetry install --all-extras
poetry shell
```

### Code Quality

Run linters and formatters:
```bash
# Format and lint Python code
poetry run ruff format .
poetry run ruff check . --fix

# Type checking
poetry run mypy src tests

# Run tests
poetry run pytest
```

Install pre-commit hooks for automatic checking:
```bash
poetry run pre-commit install
```

### Standards

- Line length: 88 characters
- Type hints required
- NumPy-style docstrings
- Conventional commit messages

Use `poetry run cz commit` for guided commit message creation.
