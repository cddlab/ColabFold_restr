"""Input isolation, novice controls, and optional-dependency boundaries."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from colabfold.rgi import SUPPORTED_MODELS, require_supported_model
from colabfold.rgi.notebook import chain_table, name_job, prepare_input
from colabfold.rgi.runtime import split_input

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rgi_toolkit") is None,
    reason="The optional RGI tests need rgi-toolkit and Python 3.12 or newer.",
)


@pytest.fixture
def raw():
    return {
        "name": "demo",
        "modelSeeds": [1],
        "sequences": [
            {"protein": {"id": ["A", "B"], "sequence": "ACDEFG"}},
            {"ligand": {"id": "C", "smiles": "CCO"}},
        ],
    }


def test_every_supported_model_accepts_optional_rgi(raw):
    config = {"conformer_restraints_config": {}}
    for model in SUPPORTED_MODELS:
        data = copy.deepcopy(raw)
        if model.startswith("esmfold2"):
            data["sequences"][0]["protein"]["id"] = "A"
        guided = prepare_input(data, model, use_rgi=True, config=config)
        assert guided["sequences"][-1]["ligand"]["conformer_restraints"] is True
        assert "conformer_restraints" not in guided["sequences"][0]["protein"]
        vanilla = prepare_input(guided, model, use_rgi=False, config=config)
        assert "restraints_config" not in vanilla
        assert all(
            "conformer_restraints" not in next(iter(x.values()))
            for x in vanilla["sequences"]
        )
    assert "restraints_config" not in raw


def test_unsupported_models_keep_vanilla(raw):
    for model in ("af2_ptm", "af2_multimer", "openbind0", "intellifold2", "unknown"):
        prepare_input(raw, model)
        with pytest.raises(ValueError, match="turn use_rgi off"):
            require_supported_model(model)


def test_conformer_controls_do_not_silently_do_nothing(raw):
    config = {"conformer_restraints_config": {}}
    for chains in ("", "Z", "C,Z"):
        with pytest.raises(ValueError):
            prepare_input(
                raw, "boltz2", use_rgi=True, config=config, conformer_chains=chains
            )
    raw["sequences"].pop()
    with pytest.raises(ValueError):
        prepare_input(raw, "boltz2", use_rgi=True, config=config)


def test_chain_table_and_partial_homomer_opt_in(raw):
    assert [r["chain"] for r in chain_table(raw)] == ["A", "B", "C"]
    guided = prepare_input(
        raw,
        "boltz2",
        use_rgi=True,
        config={"conformer_restraints_config": {}},
        conformer_chains="B",
    )
    assert "conformer_restraints" not in guided["sequences"][0]["protein"]
    assert guided["sequences"][1]["protein"]["conformer_restraints"] is True


def test_result_names_change_with_effective_settings(raw):
    def name(data=raw, model="boltz2", recycles=1):
        return name_job(
            data, model, base="demo", num_recycles=recycles, num_diffusion_samples=1
        )

    changed = copy.deepcopy(raw)
    changed["restraints_config"] = {"conformer_restraints_config": {}}
    assert len({name(), name(changed), name(model="openfold3"), name(recycles=2)}) == 4


def test_parser_removes_only_rgi_extensions_and_keeps_flags_separate(raw):
    raw = prepare_input(
        raw, "boltz2", use_rgi=True, config={"conformer_restraints_config": {}}
    )
    clean, config, flags = split_input(raw)
    assert config == {"conformer_restraints_config": {}}
    assert flags == {"A": False, "B": False, "C": True}
    assert "restraints_config" not in clean
    assert "conformer_restraints" not in clean["sequences"][-1]["ligand"]
    assert "restraints_config" in raw


def test_esm_complex_error_is_actionable(raw):
    with pytest.raises(ValueError, match="exactly one protein chain"):
        prepare_input(raw, "esmfold2")


def test_boltz_yaml_preserves_molecules_and_msa_paths():
    from colabfold.rgi.boltz import notebook_input

    entries = [
        (">A|protein|msa/query.a3m", "ACDE"),
        (">B|smiles", "CCO"),
        (">C|DNA", "ACGT"),
    ]
    config = {"conformer_restraints_config": {}}
    guided = notebook_input(entries, config=config)
    assert guided["sequences"][0]["protein"]["msa"] == "msa/query.a3m"
    assert guided["sequences"][1]["ligand"]["smiles"] == "CCO"
    assert guided["sequences"][1]["ligand"]["conformer_restraints"] is True
    assert guided["sequences"][2]["dna"]["sequence"] == "ACGT"
    vanilla = notebook_input(entries, single_sequence=True)
    assert "restraints_config" not in vanilla
    assert vanilla["sequences"][0]["protein"]["msa"] == "empty"


def test_vanilla_does_not_import_the_optional_engine(raw, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("rgi_toolkit"):
            raise ImportError("Optional engine is deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert "restraints_config" not in prepare_input(raw, "boltz2")


def test_notebooks_have_no_outputs_and_python_cells_compile():
    root = Path(__file__).resolve().parents[1]
    for name in ("ColabFold2_preview.ipynb", "AlphaFold3_of3.ipynb", "Boltz1.ipynb"):
        notebook = json.loads((root / name).read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                compile("".join(cell["source"]), name, "exec")


@pytest.mark.parametrize(
    "name", ["ColabFold2_preview.ipynb", "AlphaFold3_of3.ipynb", "Boltz1.ipynb"]
)
def test_notebook_form_requires_entries_before_prediction(name, raw):
    pytest.importorskip("ipywidgets")
    from rgi_toolkit.notebook_widgets import read_editor

    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / name).read_text())["cells"]
    cell = next(c for c in cells if c.get("metadata", {}).get("id") == "rgi-restraints")
    source = "".join(cell["source"])
    with pytest.raises(RuntimeError, match="preceding numbered"):
        exec(source, {})
    namespace = {
        "use_rgi": True,
        "fold_input": raw,
        "fasta_entries": [(">A|protein|empty", "ACDEFG")],
        "msa_mode": "single_sequence",
    }
    exec(source, namespace)
    editor = namespace["rgi_editor"]
    with pytest.raises(ValueError, match="No RGI restraints"):
        read_editor(editor)
    editor.add_buttons["distance"].click()
    editor.cards[0].fields["target_distance"].value = 30
    prediction = next(
        "".join(c["source"])
        for c in cells
        if "# Read live toolkit controls" in "".join(c["source"])
    )
    prefix = prediction.split(
        "# Rebuild the input" if name != "Boltz1.ipynb" else "# Use YAML"
    )[0]
    exec(prefix, namespace)
    assert (
        namespace["rgi_config"]["distance_restraints_config"][0]["harmonic"][
            "target_distance"
        ]
        == 30
    )
    namespace["use_rgi"] = False
    editor.add("RMSD")  # An incomplete entry must not block vanilla.
    exec(prefix, namespace)
    assert namespace["rgi_config"] is None
