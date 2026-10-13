"""Record the software environment and a checksum of every source file.

Run: python env_info.py   Writes results/environment.json.
"""
import hashlib
import json
import os
import platform
import sys

import matplotlib
import mpmath
import numpy
import scipy

SOURCES = ["nogain.py", "codes.py", "reproduce.py", "make_figures.py", "certify.py",
           "crosscheck_symplectic.py", "validate.py", "dm_check.py", "env_info.py", "run_all.sh"]
os.makedirs("results", exist_ok=True)
info = {
    "python": sys.version.split()[0],
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy": numpy.__version__, "scipy": scipy.__version__,
    "mpmath": mpmath.__version__, "matplotlib": matplotlib.__version__,
    "sha256": {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in SOURCES},
}
with open(os.path.join("results", "environment.json"), "w") as f:
    json.dump(info, f, indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "sha256"}, indent=1))
