"""Build GitHub Pages frames from actual Navier-Stokes solver output."""
import json
from pathlib import Path
from solver import simulate

OUTPUT = Path(__file__).parent / "site" / "data"

def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, shear in (("nonlinear", False), ("shear", True)):
        result = simulate(grid=32, until=0.04, dt=0.0002, shear=shear,
                          snapshot_every=10)
        frames = result.pop("frames")
        assert len(frames) == 21
        assert all(frame["max_divergence"] < 1e-8 for frame in frames)
        assert all(len(frame["vorticity"]) == 32 for frame in frames)
        (OUTPUT / (name + ".json")).write_text(
            json.dumps({"metadata": result, "frames": frames}, allow_nan=False,
                       separators=(",", ":")), encoding="utf-8")
        print(name, len(frames), "frames", frames[-1]["energy"])

if __name__ == "__main__":
    build()
