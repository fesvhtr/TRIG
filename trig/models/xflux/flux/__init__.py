try:
    from ._version import version as __version__  # type: ignore
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent
