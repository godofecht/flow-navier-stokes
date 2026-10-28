# Flow Navier–Stokes

Executable two-dimensional periodic incompressible Navier–Stokes solver in Python/NumPy, with a mathematical Flow-style equation declaration, pressure projection, and numerical verification in GitHub Actions.

## Run

```bash
python -m pip install -r requirements.txt
python solver.py --grid 32 --until 0.01 --dt 0.0001
python -m unittest discover -s tests -v
```

The solver uses second-order Runge–Kutta advection/diffusion and a conjugate-gradient pressure projection using matched forward-gradient/backward-divergence finite differences. The canonical divergence-free shear mode has the analytic continuum energy decay `exp(-2ν(2π)²t)`; the test suite validates spatial convergence against it. This is a standalone solver, **not** an upstream Flow 2.0 compiler integration.

The model is documented in `examples/navier_stokes.flow`; the executable runtime is `solver.py`. CI runs the real simulation and analytic numerical checks on Ubuntu and macOS.
