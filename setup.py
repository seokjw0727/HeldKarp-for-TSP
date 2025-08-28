from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

exts = [
    Extension(
        "hk_native",
        ["hk_native.pyx"],
        extra_compile_args=["-O3"], 
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="hk_cython_native",
    ext_modules=cythonize(
        exts,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "infer_types": True,
        },
        annotate=True,
    ),
    zip_safe=False,
)