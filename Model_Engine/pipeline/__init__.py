"""
pipeline/__init__.py
====================
Package initialiser. Runs the NumPy compatibility shim automatically
on every `import pipeline` so no other module needs to worry about it.
"""

import numpy as np

# NumPy compatibility shim — aliases removed in NumPy 1.24+
for _old, _new in [
    ('float_',   np.float64),
    ('int_',     np.int64),
    ('bool_',    np.bool_),
    ('complex_', np.complex128),
    ('unicode_', np.str_),
]:
    if not hasattr(np, _old):
        setattr(np, _old, _new)
