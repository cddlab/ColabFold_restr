"""Cover the shared-memory failure on SM 12 and previously patched installs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabfold.tokamax_compat import patch_gpu_support


@pytest.mark.parametrize(
    "gate",
    [
        "return float(device.compute_capability) >= 8.0",
        "cc = float(device.compute_capability)\n  return cc == 8.0 or cc >= 9.0",
    ],
)
def test_consumer_gpus_fall_back_and_datacenter_gpus_keep_triton(
    gate, tmp_path, monkeypatch
):
    module = tmp_path / "_src" / "gpu_utils.py"
    module.parent.mkdir()
    module.write_text("def has_triton_support(device):\n  " + gate + "\n")
    monkeypatch.setitem(
        sys.modules, "tokamax", SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    )
    platform = tmp_path / "model" / "components" / "platform.py"
    platform.parent.mkdir(parents=True)
    platform.write_text(
        "def is_datacenter_gpu(cap):\n  return cap == 8.0 or cap >= 9.0\n"
    )
    monkeypatch.setitem(
        sys.modules,
        "alphafold3",
        SimpleNamespace(__file__=str(tmp_path / "__init__.py")),
    )
    assert patch_gpu_support()
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    supported = namespace["has_triton_support"]
    exec(compile(platform.read_text(), str(platform), "exec"), namespace)
    datacenter = namespace["is_datacenter_gpu"]
    for capability in (7.0, 7.5, 8.6, 8.9, 12.0, 12.1):
        assert not supported(SimpleNamespace(compute_capability=capability))
        assert not datacenter(capability)
    for capability in (8.0, 9.0, 10.0, 10.3):
        assert supported(SimpleNamespace(compute_capability=capability))
        assert datacenter(capability)
    assert not patch_gpu_support()


@pytest.mark.parametrize(
    "notebook", ["ColabFold2_preview.ipynb", "AlphaFold3_of3.ipynb"]
)
def test_notebook_installation_applies_the_shared_gpu_patch(notebook, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / notebook).read_text())["cells"]
    installation = next(
        "".join(c["source"]) for c in cells if c["metadata"]["id"] == "install"
    )
    block = installation.split(
        "# Keep Tokamax kernels within the GPU's shared-memory limit.\n", 1
    )[1].split("# Weights, fetched", 1)[0]
    calls = []
    monkeypatch.setattr(
        "colabfold.tokamax_compat.patch_gpu_support", lambda: calls.append(True)
    )
    exec(block, {})
    assert calls == [True]
