from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name="cpe",
    version="0.0.1",
    description="Mesh Coordinate-Pair-Encoding Tokenizer",
    ext_modules=[
        Pybind11Extension(
            "_cpe",
            ["src/cpe.cpp"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "pybind11"],
)