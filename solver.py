#!/usr/bin/env python3
"""Periodic 2D incompressible Navier–Stokes with matched discrete projection."""
import argparse
import json
import math

import numpy as np


def simulate(grid=32, until=0.01, dt=0.0001, viscosity=0.001, density=1.0, shear=False):
    if not isinstance(grid, int) or not 8 <= grid <= 512:
        raise ValueError("grid must be an integer in [8, 512]")
    if not all(math.isfinite(z) for z in (until, dt, viscosity, density)):
        raise ValueError("parameters must be finite")
    if until <= 0 or dt <= 0 or viscosity < 0 or density <= 0:
        raise ValueError("invalid physical parameters")
    steps = math.ceil(until / dt)
    if not 1 <= steps <= 200000:
        raise ValueError("integration step count out of range")
    dt = until / steps
    h = 1.0 / grid
    coordinates = (np.arange(grid) + 0.5) * h
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    u = np.sin(2 * np.pi * y)
    v = np.zeros_like(u) if shear else np.sin(2 * np.pi * x)
    if dt * max((np.max(abs(u)) + np.max(abs(v))) / h,
                4 * viscosity / h**2) > 0.8:
        raise ValueError("time step fails conservative stability bound")

    def forward(a, axis):
        return (np.roll(a, -1, axis=axis) - a) / h

    def backward(a, axis):
        return (a - np.roll(a, 1, axis=axis)) / h

    def center(a, axis):
        return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2*h)

    def laplace(a):
        return (np.roll(a, -1, axis=0) + np.roll(a, 1, axis=0) +
                np.roll(a, -1, axis=1) + np.roll(a, 1, axis=1) - 4*a) / h**2

    def divergence(a, b):
        return backward(a, 0) + backward(b, 1)

    def pressure_solve(rhs):
        residual = -(rhs - rhs.mean())
        pressure = np.zeros_like(residual)
        direction = residual.copy()
        rr = float(np.sum(residual * residual))
        initial = math.sqrt(rr)
        if initial == 0:
            return pressure
        for _ in range(min(8*grid, 5000)):
            applied = -laplace(direction)
            denominator = float(np.sum(direction * applied))
            if denominator <= 0:
                raise RuntimeError("pressure operator lost positive definiteness")
            alpha = rr / denominator
            pressure += alpha * direction
            residual -= alpha * applied
            next_rr = float(np.sum(residual * residual))
            if math.sqrt(next_rr) <= 1e-10 * initial:
                return pressure - pressure.mean()
            direction = residual + (next_rr / rr) * direction
            rr = next_rr
        raise RuntimeError("pressure conjugate gradient did not converge")

    def project(a, b):
        pressure = pressure_solve(density * divergence(a, b) / dt)
        return a - dt * forward(pressure, 0) / density, b - dt * forward(pressure, 1) / density, pressure

    def rhs(a, b):
        return (-a * center(a, 0) - b * center(a, 1) + viscosity * laplace(a),
                -a * center(b, 0) - b * center(b, 1) + viscosity * laplace(b))

    u, v, pressure = project(u, v)
    energy_initial = float(0.5 * density * h*h * np.sum(u*u + v*v))
    for _ in range(steps):
        du, dv = rhs(u, v)
        stage_u, stage_v, _ = project(u + dt * du, v + dt * dv)
        du2, dv2 = rhs(stage_u, stage_v)
        u, v, pressure = project(u + dt/2 * (du + du2), v + dt/2 * (dv + dv2))
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise RuntimeError("non-finite solution")

    return {
        "grid": [grid, grid], "steps": steps, "time": until,
        "initial_kinetic_energy": energy_initial,
        "kinetic_energy": float(0.5 * density * h*h * np.sum(u*u + v*v)),
        "max_divergence": float(np.max(abs(divergence(u, v)))),
        "pressure_mean": float(pressure.mean()),
        "vorticity_rms": float(np.sqrt(np.mean((center(v, 0) - center(u, 1))**2))),
        "shear": shear,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--until", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.0001)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--shear", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps({"ok": True, **simulate(args.grid, args.until, args.dt,
                            args.viscosity, shear=args.shear)}, sort_keys=True, allow_nan=False))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Simulation failed: {error}\n")


if __name__ == "__main__":
    main()
