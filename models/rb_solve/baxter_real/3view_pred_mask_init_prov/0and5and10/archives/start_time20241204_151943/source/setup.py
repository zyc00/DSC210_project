import glob
import os

# import torch
from setuptools import find_packages
from setuptools import setup
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension
#
# requirements = ["torch", "torchvision"]

setup(
    name="crc",
    version="0.1",
    author="chenlinghao",
    packages=find_packages(exclude=("configs", "tests",)),
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)