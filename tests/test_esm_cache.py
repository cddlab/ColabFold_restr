"""Verify bounded record retention and parity with the pinned upstream cache."""

import builtins
import weakref

import numpy as np
import pytest

from colabfold.esm_cache import _write_streamed_cache, prepare_esm_cache


def test_streaming_releases_previous_weight_records(tmp_path):
    references = []

    def records():
        for index in range(8):
            assert sum(ref() is not None for ref in references) <= 1
            array = np.full((16, 32), index, dtype=np.int8)
            references.append(weakref.ref(array))
            yield f"esmc/blocks/qkv/{index}", "weights", array

    cache = _write_streamed_cache(records(), tmp_path / "cache", "esmc")
    result = np.load(cache / "blocks__qkv__weights.npy", mmap_mode="r")
    assert result.dtype == np.int8 and result.shape == (8, 16, 32)
    for index in range(8):
        np.testing.assert_array_equal(result[index], np.full((16, 32), index))
    assert not (cache / "records").exists()


def test_failed_stream_preserves_existing_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "MANIFEST").write_text("existing")

    def records():
        yield "esmc/embed", "weights", np.ones((4, 8), np.float32)
        raise RuntimeError("Truncated weight stream")

    with pytest.raises(RuntimeError, match="Truncated"):
        _write_streamed_cache(records(), cache, "esmc")
    assert (cache / "MANIFEST").read_text() == "existing"
    assert list(tmp_path.iterdir()) == [cache]


def test_unrelated_models_and_disabled_cache_keep_upstream_behavior(monkeypatch):
    real_import = builtins.__import__

    def without_predictor(name, *args, **kwargs):
        if name.startswith("alphafold3") or name.startswith("rgi_toolkit"):
            raise AssertionError("Unrelated models must not load ESM dependencies")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_predictor)
    for model in (
        "openbind0",
        "openfold3",
        "chai1",
        "esmfold2_lm300m",
        "esmfold2_lm600m",
    ):
        assert prepare_esm_cache(model) is None
    monkeypatch.setenv("AF3_ESM_CACHE", "0")
    assert prepare_esm_cache("esmfold2") is None


def test_cache_values_and_dtypes_match_upstream_loader(tmp_path, monkeypatch):
    pytest.importorskip("alphafold3.model.esm")
    import zstandard
    from alphafold3.model import esm, params

    rng = np.random.default_rng(42)
    records = [
        ("__meta__", "version", np.array([1], np.int32)),
        ("esmc/embed", "weights", rng.integers(-127, 128, (33, 64), dtype=np.int8)),
        ("esmc/embed", "weights__q_scale", rng.random(64).astype(np.float32)),
        ("esmc/final_ln", "scale", rng.random(64).astype(np.float16)),
        (
            "esmc/blocks/prestacked",
            "weights",
            rng.integers(-127, 128, (2, 64, 64), dtype=np.int8),
        ),
        (
            "esmc/blocks/prestacked",
            "weights__q_scale",
            rng.random((2, 64)).astype(np.float32),
        ),
    ]
    for layer in (1, 0):
        records.extend(
            [
                (
                    f"esmc/blocks/qkv/{layer}",
                    "weights",
                    rng.integers(-127, 128, (64, 192), dtype=np.int8),
                ),
                (
                    f"esmc/blocks/qkv/{layer}",
                    "weights__q_scale",
                    rng.random(192).astype(np.float16),
                ),
                (
                    f"esmc/blocks/pre_norm/{layer}",
                    "scale",
                    rng.random(64).astype(np.float16),
                ),
            ]
        )
    blob = tmp_path / "esmc.bin.zst"
    blob.write_bytes(
        zstandard.ZstdCompressor().compress(
            b"".join(params.encode_record(*record) for record in records)
        )
    )
    monkeypatch.setenv("AF3_ESM_CACHE", "0")
    monkeypatch.setenv("AF3_ESM_DEVICE_CACHE", "0")
    expected, expected_dims = esm.load(tmp_path, "esmc", "esmc")
    with blob.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            cache = _write_streamed_cache(
                params.read_records(stream), tmp_path / "esmc.unpacked", "esmc"
            )
    actual = esm._read_cache(str(cache))
    assert set(actual) == set(expected)
    assert esm._dims_from(actual, "esmc") == expected_dims
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
