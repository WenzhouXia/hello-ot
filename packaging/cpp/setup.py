from setuptools import Extension, setup
import pybind11


setup(ext_modules=[Extension(
    "hello_ot_cpp." + module, ["native_src/" + source + ".cpp"],
    include_dirs=[pybind11.get_include()], language="c++",
    extra_compile_args=["-O3", "-std=c++17", "-fopenmp", "-ffp-contract=off"],
    extra_link_args=["-fopenmp"],
) for module, source in [("_lp", "solver"), ("_scan", "scan")]])
