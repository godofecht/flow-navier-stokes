"""Analytic solution and discrete projection acceptance."""
import math
import unittest

from solver import simulate


class SolverTest(unittest.TestCase):
    def test_incompressible_periodic_shear(self):
        result = simulate(grid=32, shear=True)
        self.assertEqual(result["steps"], 100)
        self.assertLess(result["max_divergence"], 1e-9)
        self.assertAlmostEqual(result["pressure_mean"], 0, places=9)
        target = math.exp(-2 * 0.001 * (2 * math.pi)**2 * 0.01)
        self.assertLess(abs(result["kinetic_energy"] / result["initial_kinetic_energy"] - target), 3e-6)

    def test_refinement_approaches_analytic_decay(self):
        errors = []
        target = math.exp(-2 * 0.001 * (2 * math.pi)**2 * 0.01)
        for grid in (16, 32, 64):
            result = simulate(grid=grid, shear=True)
            errors.append(abs(result["kinetic_energy"] / result["initial_kinetic_energy"] - target))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_nonlinear_initial_condition_projection(self):
        result = simulate(grid=32, shear=False)
        self.assertLess(result["max_divergence"], 1e-8)
        self.assertGreater(result["kinetic_energy"], 0)
        self.assertLessEqual(result["kinetic_energy"], result["initial_kinetic_energy"] * 1.001)

    def test_rejects_invalid_inputs(self):
        for settings in ({"grid": 4}, {"until": -1}, {"dt": 0},
                         {"viscosity": -1}, {"density": 0},
                         {"until": float("nan")}, {"dt": 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                simulate(**settings)


if __name__ == "__main__":
    unittest.main()
