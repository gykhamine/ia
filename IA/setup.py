"""
IA/setup.py — Fallback setuptools (optionnel, make est la methode principale).

Compilation principale:
    cd IA/
    make          # compile _ia_core.so (g++ pur, ni CMake ni pybind11)

Fallback (si vous preferez pip):
    pip install -e .
"""

from setuptools import setup, Extension
import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Extension(
        name="IA.cpp._ia_core",
        sources=[os.path.join("cpp", "c_api.cpp"), os.path.join("cpp", "engine.cpp")],
        include_dirs=[os.path.join(pkg_dir, "cpp")],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-march=native", "-ffast-math"],
    ),
]

setup(
    name="IA",
    version="1.0.0",
    description="Module IA avec moteur C++ (make, sans CMake ni pybind11)",
    packages=["IA", "IA.train", "IA.infer", "IA.cpp"],
    package_dir={"IA": pkg_dir},
    ext_modules=ext_modules,
    install_requires=["numpy>=1.21"],
    python_requires=">=3.9",
)