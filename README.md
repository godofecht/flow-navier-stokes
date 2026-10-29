# Flow Navier–Stokes

Executable two-dimensional periodic incompressible Navier–Stokes solver in Python/NumPy, with a mathematical Flow-style equation declaration, pressure projection, and numerical verification in GitHub Actions.

## Run

```bash
python -m pip install -r requirements.txt
python solver.py --grid 32 --until 0.01 --dt 0.0001
python -m unittest discover -s tests -v
```

The solver uses second-order Runge–Kutta advection/diffusion and a conjugate-gradient pressure projection using matched forward-gradient/backward-divergence finite differences. The canonical divergence-free shear mode has the analytic continuum energy decay `exp(-2ν(2π)²t)`; the test suite validates spatial convergence against it. This is a standalone solver, **not** an upstream Flow 2.0 compiler integration.

The checked-in `examples/navier_stokes.flow` is now **executable** by FlowStudio's separate mathematical Flow frontend, while `solver.py` remains a standalone NumPy implementation. CI checks out the exact pinned FlowStudio compiler commit `97992dd98d5c260293201d4a4bb2009b8ab28046`, compiles and executes this repository's actual `.flow` file, compares its numerical output against the standalone solver, and verifies analytic shear-mode decay on Ubuntu and macOS. This is not upstream Flow 2.0 compiler support.

To execute the `.flow` source locally, check out that FlowStudio commit into `_flowstudio` and run:

```sh
git clone https://github.com/godofecht/FlowStudio.git _flowstudio
git -C _flowstudio checkout 97992dd98d5c260293201d4a4bb2009b8ab28046
python _flowstudio/src/math_flow.py --check examples/navier_stokes.flow
python _flowstudio/src/math_flow.py --grid 16 --until 0.001 examples/navier_stokes.flow
```

The full declared `32 × 32` grid and `0.01s` duration execute without overrides, too. Both `solver.py --trace` and the checked-in `.flow` program now emit measured first-step RK2 semantics. The FlowStudio mathematical frontend additionally exposes pressure-projection diagnostics and the native workbench can display them. The two trace schemas are intentionally distinct.
