# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by AscendSpeed.")

__description__ = 'AscendSpeed for LLMs of Ascend'
__version__ = '0.0.1'
__author__ = 'Ascend'
__long_description__ = 'AscendSpeed for LLMs of Ascend'
__url__ = 'https://gitee.com/ascend/AscendSpeed'
__download_url__ = 'https://gitee.com/ascend/AscendSpeed/release'
__keywords__ = 'Ascend, langauge, deep learning, NLP'
__license__ = 'See https://gitee.com/ascend/AscendSpeed'
__package_name__ = 'ascendspeed'
__contact_names__ = 'Ascend'


with open("README.md", "r") as fh:
    long_description = fh.read()

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")
cmd_class={}
exts = []
try:
    from torch_npu.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension
    import imp
    ext_ops = cpp_extension.NpuExtension(
        name="ascendspeed_ops_internel",
        sources=["ascendspeed/ops/FlashAttention.cpp"],
        extra_compile_args=[
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
        '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
        '-I'+imp.find_module('torch_npu')[1]+'/include/third_party/acl/inc'
    ],
    )
    exts.append(ext_ops)
    cmd_class={"build_ext": BuildExtension.with_options(use_ninja=False)}
except Exception:
    print('Can not find any torch_npu, ops setup failed')

setuptools.setup(
    package_data={'ascendspeed':['ascendspeed/data/Makefile']},
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=__url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass=cmd_class,
    ext_modules=exts
)
