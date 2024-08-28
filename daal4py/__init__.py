# ==============================================================================
# Copyright 2014 Intel Corporation
# Copyright 2024 Fujitsu Limited
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
# ==============================================================================

import platform

if "Windows" in platform.system():
    import os
    import site
    import sys

    arch_dir = platform.machine()
    print(arch_dir)
    plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
    arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir

    current_path = os.path.dirname(__file__)
    path_to_env = site.getsitepackages()[0]
    print(path_to_env)
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    print(path_to_libs)
    print(sys.version_info.minor)
    if sys.version_info.minor >= 8:
        if "DALROOT" in os.environ:
            dal_root_redist = os.path.join(os.environ["DALROOT"], "redist", arch_dir)
            print(dal_root_redist)
            if os.path.exists(dal_root_redist):
                os.add_dll_directory(dal_root_redist)
                print("is os path exist")
                os.environ["PATH"] = dal_root_redist + os.pathsep + os.environ["PATH"]
                print(os.environ["PATH"])
        os.add_dll_directory(path_to_libs)
    os.environ["PATH"] = path_to_libs + os.pathsep + os.environ["PATH"]

try:
    from daal4py._daal4py import *
    from daal4py._daal4py import (
        __has_dist__,
        _get__daal_link_version__,
        _get__daal_run_version__,
        _get__version__,
    )
except ImportError as e:
    s = str(e)
    if "libfabric" in s:
        raise ImportError(
            s + "\n\nActivating your conda environment or sourcing mpivars."
            "[c]sh/psxevars.[c]sh may solve the issue.\n"
        )

    raise

from . import mb, sklearn

__all__ = ["mb", "sklearn"]
