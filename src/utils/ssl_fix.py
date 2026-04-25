"""SSL certificate fix for HuggingFace Hub downloads.

This module patches httpx to disable SSL verification, which is needed
on some macOS systems where the Python SSL certificates are not properly
configured.

Usage:
    import src.utils.ssl_fix  # Import at the very top of your script

WARNING: This disables SSL verification. Only use in development environments.
"""

import ssl
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create an unverified SSL context for stdlib
ssl._create_default_https_context = ssl._create_unverified_context

# Patch httpx to disable SSL verification
try:
    import httpx

    _original_client_init = httpx.Client.__init__
    _original_async_client_init = httpx.AsyncClient.__init__

    def _patched_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        _original_client_init(self, *args, **kwargs)

    def _patched_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        _original_async_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client_init
    httpx.AsyncClient.__init__ = _patched_async_client_init

except ImportError:
    pass

# Also set environment variables as fallback
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

