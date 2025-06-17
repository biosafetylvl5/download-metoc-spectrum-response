"""Download METOC Spectrum Response package for downloading SRF files."""

import logging

try:
    from importlib.metadata import version

    __version__ = version("download-metoc-spectrum-response")
except ImportError:
    try:
        from importlib_metadata import version

        __version__ = version("download-metoc-spectrum-response")
    except ImportError:
        logging.debug(
            "Could not set __version__ because importlib.metadata is not available. "
            "If running python 3.7, installing importlib-metadata will fix this issue",
        )
