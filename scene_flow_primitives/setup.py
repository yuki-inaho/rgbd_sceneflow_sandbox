"""Build a C ABI shared library; ctypes loads it without a Python C-API dependency."""
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CAbiBuildExt(build_ext):
    """Do not ask the linker for a PyInit symbol: this library is loaded by ctypes."""
    def get_export_symbols(self, ext):
        return ext.export_symbols
flags = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3", "-Wall", "-Wextra"]
setup(cmdclass={"build_ext":CAbiBuildExt},ext_modules=[Extension("scene_flow_refinement._native",
    sources=["cpp/bindings_c.cpp", "cpp/matcher.cpp", "cpp/solver.cpp", "cpp/renderer.cpp", "cpp/rigid.cpp"],
    language="c++", extra_compile_args=flags)])
