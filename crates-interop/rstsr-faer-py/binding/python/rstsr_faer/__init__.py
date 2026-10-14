"""rstsr_faer — Python array-API binding of rstsr (faer device).

Thin package: the native layer is ``rstsr_faer.rstsr_faer`` (pyo3), and the
array-API-graded namespace is :mod:`rstsr_faer.api`.  No algorithms live on
the Python side; Python only shuffles handles and mirrors signatures.
"""

from . import rstsr_faer as _native
from . import api

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("rstsr-faer-py")
except Exception:  # not installed (running from the source tree)
    __version__ = "0.0.0"
