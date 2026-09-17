"""Genesis evaluation suite.

Importing this package adds the python/ source directory to sys.path
so that eval modules can import genesis_cognitive and genesis_client
without depending on import order.
"""

import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
