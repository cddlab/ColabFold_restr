"""Align predictor and Tokamax dispatch with consumer GPU shared-memory limits."""

from pathlib import Path


def patch_gpu_support():
    """Update the installed gate for predictor subprocesses, including reruns."""
    import alphafold3
    import tokamax

    triton_condition = "return cc == 8.0 or 9.0 <= cc < 12.0"
    predictor_condition = "return cap == 8.0 or 9.0 <= cap < 12.0"
    gates = [
        (
            Path(tokamax.__file__).parent / "_src" / "gpu_utils.py",
            {
                "return float(device.compute_capability) >= 8.0": (
                    "cc = float(device.compute_capability)\n  " + triton_condition
                ),
                "return cc == 8.0 or cc >= 9.0": triton_condition,
            },
        ),
        (
            Path(alphafold3.__file__).parent / "model" / "components" / "platform.py",
            {"return cap == 8.0 or cap >= 9.0": predictor_condition},
        ),
    ]
    changed = False
    for path, replacements in gates:
        source = path.read_text()
        updated = source
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != source:
            path.write_text(updated)
            changed = True
    return changed
