# scikit-learn-intelex - Claude Code Guidelines

## Current Work

- **Task**: HDBSCAN algorithm integration (oneDAL -> sklearnex patching)
- **Status**: Implementation complete — C++ pybind11 wrapper, onedal Python layer, sklearnex patching, tests. Pending build & validation.
- **Dependencies**: oneDAL HDBSCAN from branch `dev/asolovev_hdbscan_ai` in `/localdisk2/mkl/asolovev/oneDAL`
- **Last updated**: 2026-04-23

## Goal: Accelerate sklearn with oneDAL

scikit-learn-intelex patches sklearn estimators with Intel oneDAL-backed implementations. The goal is:
1. **Drop-in replacement** — `from sklearnex import patch_sklearn; patch_sklearn()` makes sklearn use oneDAL
2. **Correctness first** — Results must match sklearn exactly (ARI=1.0 for clustering)
3. **Performance** — 10-100x speedup over stock sklearn on CPU, even more on GPU
4. **Graceful fallback** — Unsupported parameters/data types fall back to sklearn transparently

## Self-Research Policy

Before starting any task, proactively explore the relevant parts of the codebase to understand context and patterns. After completing any task, update the "Recent Changes Log" section at the bottom.

## Architecture (4 Layers)

```
User Code -> sklearnex/  (sklearn-compatible API, patching, dispatch)
                  |
                  v
             onedal/     (Python wrapper around C++ backend)
                  |
                  v
             onedal/*.cpp (pybind11 bindings to oneDAL C++)
                  |
                  v
             Intel oneDAL C++ library (the actual compute)
```

### Adding a New Algorithm (Checklist)

Using DBSCAN as template, a new algorithm needs:

| Layer | File | Purpose |
|-------|------|---------|
| C++ pybind11 | `onedal/cluster/<algo>.cpp` | Binds C++ descriptor/compute to Python |
| C++ registration | `onedal/dal.cpp` | `ONEDAL_PY_INIT_MODULE(<algo>)` + `init_<algo>(m)` in all 4 places (SPMD decl, non-SPMD decl, SPMD init, non-SPMD init) |
| onedal Python | `onedal/cluster/<algo>.py` | Low-level wrapper: `_get_onedal_params()`, `fit()`, `compute()` |
| onedal __init__ | `onedal/cluster/__init__.py` | Export the class |
| sklearnex Python | `sklearnex/cluster/<algo>.py` | sklearn-compatible class with `_onedal_fit()`, `_onedal_supported()`, `fit()` |
| sklearnex __init__ | `sklearnex/cluster/__init__.py` | Export the class |
| Dispatcher | `sklearnex/dispatcher.py` | Add to `get_patch_map_core()` — import + mapping entry |
| Tests | `sklearnex/cluster/tests/test_<algo>.py` | Parametrized tests with dataframe/queue support |
| SPMD onedal | `onedal/spmd/cluster/<algo>.py` | Thin wrapper with `bind_spmd_backend` |
| SPMD sklearnex | `sklearnex/spmd/cluster/<algo>.py` | Override `_onedal_<algo>` to use SPMD backend |
| SPMD __init__ | Both `onedal/spmd/cluster/__init__.py` and `sklearnex/spmd/cluster/__init__.py` | Export |

### Key Patterns

**Dispatch mechanism** (`sklearnex/_device_offload.py`):
- `dispatch(self, "fit", {"onedal": _onedal_fit, "sklearn": _sklearn_fit}, X, y)`
- Calls `_onedal_cpu_supported()` / `_onedal_gpu_supported()` to check if oneDAL can handle the params
- Falls back to sklearn if conditions fail

**Backend binding** (`onedal/common/_backend.py`):
- `@bind_default_backend("algo_name.task_name")` decorates `compute()` method
- Maps to pybind11 module: `backend.algo_name.task_name.compute(policy, params, ...)`

**Build system**: CMake with `GLOB_RECURSE` — any `.cpp` under `onedal/` is auto-compiled. No need to edit CMakeLists.txt.

**`dal.cpp` registration**: 4 places to add `init_<algo>`:
1. SPMD forward declarations (under `#ifdef ONEDAL_DATA_PARALLEL_SPMD`)
2. Non-SPMD forward declarations (under `#else`)
3. SPMD `PYBIND11_MODULE` body
4. Non-SPMD `PYBIND11_MODULE` body

## How to Build

### Full Build (from scratch)

```bash
cd /localdisk2/mkl/asolovev

# 1. Proxies and env
export https_proxy=http://proxy-dmz.intel.com:912
export http_proxy=http://proxy-dmz.intel.com:911
export no_proxy=intel.com,localhost,127.0.0.1
export SYCL_DEVICE_FILTER=level_zero:gpu

# 2. Source infra BEFORE conda (critical: conda python gets overwritten otherwise)
source libraries.performance.data-analytics.dal-infra/env/scripts/linux.sh distributed dpcpp

# 3. Build oneDAL
cd oneDAL
make onedal_dpc -j48
source __release_lnx/daal/latest/env/vars.sh
cd ..

# 4. Conda environment
conda create -n build_sklearnex python=3.12 -y
conda activate build_sklearnex
cd scikit-learn-intelex
pip install -r dependencies-dev

# 5. Build sklearnex
export MPIROOT=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
rm -rf build* record*.txt *.egg-info $(find . -name *.so | xargs)
python setup.py develop --no-deps

# 6. Install test deps
pip install -r requirements-test.txt
```

### Rebuild After Code Changes

```bash
# Python-only changes: just re-run
python setup.py develop --no-deps

# C++ changes (onedal/*.cpp): clean build
rm -rf build*
python setup.py develop --no-deps
```

### GPU/SPMD Dependencies (Optional)

```bash
conda install mpi4py dpctl dpnp -c https://software.repos.intel.com/python/conda --no-deps
```

### Running Tests

```bash
# Run specific test
pytest sklearnex/cluster/tests/test_hdbscan.py -v

# Run all cluster tests
pytest sklearnex/cluster/tests/ -v

# Run from tests/ directory for full suite
cd tests/
pytest
```

## Coding Conventions

- **Python**: Follow sklearn conventions — `fit()` returns self, `labels_` with trailing underscore
- **C++**: Follow oneDAL conventions (see oneDAL CLAUDE.md)
- **Tests**: Use `@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())` for device coverage
- **Imports**: `from sklearn.X import Y as _sklearn_Y` for base classes
- **Copyright**: `Copyright contributors to the oneDAL project` for new files
- **Modified files**: Keep original Intel copyright, add contributor line

## Key Paths

| What | Where |
|------|-------|
| sklearn wrappers | `sklearnex/<category>/<algo>.py` |
| Dispatcher/patching | `sklearnex/dispatcher.py` |
| Device offload | `sklearnex/_device_offload.py` |
| Patching conditions | `sklearnex/_utils.py` (PatchingConditionsChain) |
| Base class | `sklearnex/base.py` (oneDALEstimator) |
| onedal Python wrappers | `onedal/<category>/<algo>.py` |
| C++ pybind11 bindings | `onedal/<category>/<algo>.cpp` |
| Module registration | `onedal/dal.cpp` |
| Backend manager | `onedal/common/_backend.py` |
| SPMD wrappers | `onedal/spmd/` and `sklearnex/spmd/` |
| Tests | `sklearnex/<category>/tests/` |
| Build scripts | `scripts/CMakeLists.txt` |

## Lessons Learned

### Build System
- `GLOB_RECURSE` picks up all `.cpp` under `onedal/` — no CMakeLists.txt edits needed
- `python setup.py develop --no-deps` builds C++ and installs in-place
- Source infra BEFORE conda activation to avoid python version conflicts
- `dal.cpp` needs updates in 4 places for each new algorithm module

### Patching System
- `dispatcher.py::get_patch_map_core()` is LRU-cached — changes need process restart
- Each mapping entry: `(module, "ClassName", sklearnex_class, sklearn_class)`
- `None` as sklearn_class means sklearnex-only (no sklearn equivalent)

### Testing
- `get_dataframes_and_queues()` provides CPU + GPU test combinations
- `_convert_to_dataframe()` handles numpy/dpnp/pandas conversion
- Tests should verify `"sklearnex" in estimator.__module__` to confirm patching worked

## Recent Changes Log

### HDBSCAN Integration (2026-04-23)

Added HDBSCAN algorithm support through all layers of scikit-learn-intelex.

**Files created:**
- `onedal/cluster/hdbscan.cpp` — pybind11 bindings for oneDAL HDBSCAN (brute_force, kd_tree methods, 5 metrics)
- `onedal/cluster/hdbscan.py` — Low-level Python wrapper with `_get_onedal_params()`, auto method selection
- `sklearnex/cluster/hdbscan.py` — sklearn-compatible HDBSCAN with dispatch, fallback conditions
- `sklearnex/cluster/tests/test_hdbscan.py` — 7 parametrized tests (import, clustering, shape, vs-sklearn, fallback, metrics, minkowski)
- `onedal/spmd/cluster/hdbscan.py` — SPMD backend binding
- `sklearnex/spmd/cluster/hdbscan.py` — SPMD sklearnex wrapper

**Files modified:**
- `onedal/dal.cpp` — Registered `hdbscan` module in all 4 init points
- `onedal/cluster/__init__.py` — Added HDBSCAN export
- `sklearnex/cluster/__init__.py` — Added HDBSCAN export
- `sklearnex/dispatcher.py` — Added HDBSCAN to patch map
- `onedal/spmd/cluster/__init__.py` — Added HDBSCAN export
- `sklearnex/spmd/cluster/__init__.py` — Added HDBSCAN export

**Supported oneDAL features:**
- Methods: brute_force (O(N^2)), kd_tree (O(N log^2 N))
- Metrics: euclidean, manhattan, minkowski, chebyshev, cosine
- Auto method selection: kd_tree for Lp metrics, brute_force for cosine

**Fallback conditions** (falls back to sklearn when):
- Unsupported metric (not in {euclidean, manhattan, minkowski, chebyshev, cosine})
- cluster_selection_method != "eom"
- cluster_selection_epsilon != 0.0
- max_cluster_size specified
- allow_single_cluster=True
- store_centers specified
- Sparse input
