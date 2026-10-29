"""Verify exported visualization data is genuine, finite solver output."""
import json
import math
from pathlib import Path
import unittest

from solver import simulate


class SiteDataTest(unittest.TestCase):
    def test_exported_fields_match_solver(self):
        root = Path(__file__).resolve().parents[1] / "site" / "data"
        for name, shear in (("nonlinear", False), ("shear", True)):
            with self.subTest(name=name):
                payload = json.loads((root / (name + ".json")).read_text())
                frames = payload["frames"]
                self.assertEqual(len(frames), 21)
                self.assertEqual(frames[0]["step"], 0)
                self.assertEqual(frames[-1]["step"], 200)
                self.assertLess(frames[-1]["max_divergence"], 1e-8)
                for frame in frames:
                    for field in ("u", "v", "pressure", "vorticity"):
                        self.assertEqual(len(frame[field]), 32)
                        self.assertTrue(all(len(row) == 32 for row in frame[field]))
                        self.assertTrue(all(math.isfinite(v) for row in frame[field] for v in row))
                exact = simulate(grid=32, until=0.04, dt=0.0002, shear=shear)
                self.assertAlmostEqual(frames[-1]["energy"], exact["kinetic_energy"], places=12)


if __name__ == "__main__":
    unittest.main()
