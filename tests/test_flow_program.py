"""Cross-repository acceptance: execute the checked-in .flow model, not a Python stand-in."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from solver import simulate

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "_flowstudio" / "src" / "math_flow.py"
SOURCE = ROOT / "examples" / "navier_stokes.flow"


@unittest.skipUnless(FRONTEND.is_file(), "Pinned FlowStudio frontend not checked out")
class ExecutableFlowProgramTest(unittest.TestCase):
    def run_flow(self, source, *arguments):
        result = subprocess.run([sys.executable, str(FRONTEND), *arguments, str(source)],
                                cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_real_flow_compilation_and_numerical_execution(self):
        checked = self.run_flow(SOURCE, "--check")
        self.assertTrue(checked["ok"])
        self.assertEqual(checked["grid"], [32, 32])
        result = self.run_flow(SOURCE, "--grid", "16", "--until", "0.001")
        self.assertTrue(result["ok"])
        self.assertEqual(result["steps"], 10)
        self.assertEqual(result["grid"], [16, 16])
        self.assertLess(result["max_divergence"], 1e-5)
        trace = result["semantic_trace"]
        self.assertEqual((trace["schema"], trace["field"]), (2, "velocity"))
        self.assertEqual([stage["name"] for stage in trace["stages"]],
                         ["initialSlope", "predictorSlope"])
        for stage in trace["stages"]:
            self.assertEqual([term["role"] for term in stage["terms"]],
                             ["advection", "diffusion"])
            self.assertLess(stage["maximumDecompositionResidual"], 1e-10)
        self.assertEqual([entry["name"] for entry in trace["projections"]],
                         ["initial", "predictor", "corrector"])
        for entry in trace["projections"]:
            self.assertLess(entry["maximumDivergenceAfter"], 1e-5)
        self.assertLess(abs(result["pressure_mean"]), 1e-8)
        reference = simulate(grid=16, until=0.001, dt=0.0001)
        self.assertLess(abs(result["kinetic_energy"] - reference["kinetic_energy"]), 1e-3)

    def test_flow_shear_fourier_decay_and_projection(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("sin(2πx / 1m) * 1m/s", source)
        shear = source.replace("sin(2πx / 1m) * 1m/s", "0m/s")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shear.flow"
            path.write_text(shear, encoding="utf-8")
            result = self.run_flow(path, "--grid", "32", "--until", "0.01")
        target = math.exp(-2 * 0.001 * (2 * math.pi)**2 * 0.01)
        ratio = result["kinetic_energy"] / result["initial_kinetic_energy"]
        self.assertLess(abs(ratio - target), 1e-5)
        self.assertLess(result["max_divergence"], 1e-8)
        self.assertLess(result["semantic_trace"]["stages"][0]["terms"][0]
                        ["maximumMagnitude"], 1e-10)


if __name__ == "__main__":
    unittest.main()
