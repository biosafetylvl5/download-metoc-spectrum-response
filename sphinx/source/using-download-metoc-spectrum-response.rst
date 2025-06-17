======================================
Using download-metoc-spectrum-response
======================================

Download satellite instrument spectral response function files from meteorological and oceanographic instruments.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

This is a command-line tool for downloading SRF files based on YAML metadata. Supports multiple formats, filtering, and concurrent downloads.

Features:
* Download .txt, .tar.gz, and .flt files
* Filter by platform, instrument, channel, or file type
* Parallel downloads with progress tracking
* Dry run mode and resume support
* Uses built-in SRF index by default

Installation
============

With Poetry::

   git clone https://github.com/biosafetylvl5/download-metoc-spectrum-response.git
   cd download-metoc-spectrum-response
   poetry install && poetry shell

With pip::

   pip install aiohttp PyYAML typer rich

Quick Start
===========

1. Check what's available (uses built-in index)::

      srf-downloader info
      srf-downloader list-options

2. Search for files::

      srf-downloader search --platform visible_ir_sensors

3. Download::

      srf-downloader download --platform visible_ir_sensors

Commands
========

All commands use the built-in ``srf_index.yaml`` by default. Provide a custom YAML file as the first argument if needed.

info
----

Show dataset statistics::

   # Built-in index
   srf-downloader info
   
   # Custom file
   srf-downloader info srf_files_extracted.yaml

list-options
------------

Show available platforms, instruments, and file types::

   # Built-in index
   srf-downloader list-options
   
   # Custom file
   srf-downloader list-options srf_files_extracted.yaml

search
------

Find files without downloading::

   srf-downloader search [yaml-file] [OPTIONS]

Options:
* ``--platform, -p``: Platform name
* ``--instrument, -i``: Instrument name  
* ``--channel, -c``: Channel identifier
* ``--type, -t``: File type (txt, tar.gz, flt)

Examples::

   # By platform (built-in index)
   srf-downloader search --platform visible_ir_sensors
   
   # By instrument with custom file
   srf-downloader search my_data.yaml --instrument avhrr
   
   # Multiple filters
   srf-downloader search --platform visible_ir_sensors --instrument avhrr

download
--------

Download matching files::

   srf-downloader download [yaml-file] [OPTIONS]

Options:
* ``--output, -o``: Output directory (default: srf_downloads)
* ``--concurrent``: Max concurrent downloads (default: 5)
* ``--dry-run``: Preview without downloading
* Same filter options as search

Examples::

   # Basic download (built-in index)
   srf-downloader download --instrument avhrr
   
   # Custom file with location
   srf-downloader download my_data.yaml --output ./my_files --concurrent 10
   
   # Preview first
   srf-downloader download --dry-run

Common Tasks
============

Download by instrument::

   srf-downloader download --instrument avhrr
   srf-downloader download --instrument modis

Download specific channels::

   srf-downloader download --channel "01"
   srf-downloader download --instrument avhrr --channel "3.7"

Download by file type::

   # Individual channel files
   srf-downloader download --type txt
   
   # Archive files
   srf-downloader download --type tar.gz

Preview large downloads::

   srf-downloader download --platform visible_ir_sensors --dry-run

Using custom YAML files::

   srf-downloader download my_custom_srf.yaml --instrument modis
   srf-downloader search old_data.yaml --platform "NOAA-20"

File Organization
=================

Downloads are organized as::

   output_directory/
   ├── platform1/
   │   ├── instrument1/
   │   │   ├── file1.txt
   │   │   └── file2.txt
   │   └── instrument2/
   │       └── file3.tar.gz

Default YAML File
=================

The tool includes a built-in ``srf_index.yaml`` file with current SRF data. When you run commands without specifying a YAML file, it automatically uses this default.

You'll see a message like::

   Using default SRF index: /path/to/srf_index.yaml

To use a different file, provide it as the first argument::

   srf-downloader download my_custom_data.yaml --instrument avhrr

The tool falls back to ``srf_files_extracted.yaml`` if the main index isn't found.

Performance Tips
================

* Start with default concurrency (5), adjust if needed
* Use ``--dry-run`` for large downloads
* Higher concurrency for fast networks: ``--concurrent 15``
* Lower for slow/unstable connections: ``--concurrent 2``

Troubleshooting
===============

**No files found**: Check available options with ``list-options``

**Download failures**: Reduce ``--concurrent`` value, check network

**Slow downloads**: Increase ``--concurrent`` or check bandwidth

**Permission errors**: Verify write access to output directory

**Default file not found**: Provide a custom YAML file explicitly

Scripting Example
=================

::

   #!/bin/bash
   
   # Download multiple instruments using built-in index
   for instrument in "avhrr" "modis" "viirs"; do
       echo "Downloading $instrument..."
       srf-downloader download \
           --instrument "$instrument" \
           --output "./srf_data"
   done
   
   # Use custom data file
   srf-downloader download my_srf_data.yaml \
       --platform "NOAA-20" \
       --output "./noaa20_data"

The tool automatically skips existing files, so you can safely re-run downloads.
