"""Prepare the upstream ESM-C cache without retaining the complete decoded blob."""

import os
import re
import shutil
import tempfile
from pathlib import Path


def _write_streamed_cache(records, cache, family, scale_suffix="__q_scale"):
    import numpy as np

    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{cache.name}-", dir=cache.parent
    ) as temp:
        stage = Path(temp)
        parts = stage / "records"
        parts.mkdir()
        leaves, blocks = {}, {}
        for index, (scope, name, array) in enumerate(records):
            if scope.startswith("__meta__"):
                continue
            sub = scope[len(family) + 1 :] if scope.startswith(family + "/") else scope
            match = re.fullmatch(r"(blocks/.+)/(\d+)", sub)
            path = parts / f"{index}.npy"
            np.save(path, array, allow_pickle=False)
            if match:
                key, layer = f"{match[1]}/{name}", int(match[2])
                group = blocks.setdefault(key, {})
                if layer in group:
                    raise ValueError(f"Duplicate ESM parameter: {key}, layer {layer}")
                group[layer] = path
            else:
                key = f"{sub}/{name}" if sub else name
                if key in leaves:
                    raise ValueError(f"Duplicate ESM parameter: {key}")
                leaves[key] = path

        keys, consumed_scales = [], set()
        for key, path in leaves.items():
            if key in consumed_scales:
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            scale_key = key + scale_suffix
            if key.endswith(scale_suffix):
                if (
                    not key.startswith("blocks/")
                    and key[: -len(scale_suffix)] in leaves
                ):
                    continue
            elif key.startswith("blocks/"):
                if array.dtype != np.int8:
                    array = np.asarray(array, np.float32)
            elif scale_key in leaves:
                scale = np.load(leaves[scale_key], mmap_mode="r", allow_pickle=False)
                array = (
                    array.reshape(-1, array.shape[-1]).astype(np.float32) * scale
                ).reshape(array.shape)
                consumed_scales.add(scale_key)
            else:
                array = np.asarray(array, np.float32)
            np.save(
                stage / (key.replace("/", "__") + ".npy"), array, allow_pickle=False
            )
            keys.append(key)

        for key, layers in blocks.items():
            indices = sorted(layers)
            if indices != list(range(len(indices))):
                raise ValueError(f"Missing ESM layers for {key}: {indices}")
            first = np.load(layers[indices[0]], mmap_mode="r", allow_pickle=False)
            dtype = first.dtype
            if not key.endswith(scale_suffix) and dtype != np.int8:
                dtype = np.dtype(np.float32)
            output = np.lib.format.open_memmap(
                stage / (key.replace("/", "__") + ".npy"),
                mode="w+",
                dtype=dtype,
                shape=(len(indices), *first.shape),
            )
            for position, index in enumerate(indices):
                layer = np.load(layers[index], mmap_mode="r", allow_pickle=False)
                if layer.shape != first.shape or layer.dtype != first.dtype:
                    raise ValueError(f"Inconsistent ESM layer layout for {key}")
                output[position] = layer
            output.flush()
            del output, first, layer
            keys.append(key)

        if not keys:
            raise ValueError("The ESM weight stream contains no parameters.")
        shutil.rmtree(parts)
        (stage / "MANIFEST").write_text("\n".join(keys))
        if cache.exists():
            shutil.rmtree(cache)
        stage.rename(cache)
    return cache


def prepare_esm_cache(model_name):
    """Use the unchanged 6B tower with the pinned runner's memory-mapped cache."""
    if model_name != "esmfold2" or os.environ.get("AF3_ESM_CACHE", "1") == "0":
        return None

    import fcntl
    import importlib.metadata

    if importlib.metadata.version("alphafold3-colabfold") != "3.1.11":
        return None

    import zstandard
    from alphafold3.model import esm, model_registry, params, weights

    tower = model_registry.ESMFOLD2_VARIANTS[model_name]["esmc"]
    directory = Path(weights.default_dir(tower))
    directory.mkdir(parents=True, exist_ok=True)
    blob = directory / f"{tower}.bin.zst"
    cache = Path(esm._cache_dir(str(blob)))
    with (directory / f"{tower}.cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if esm._read_cache(str(cache)) is not None:
            return cache
        if not blob.exists():
            spec = model_registry.get(model_name)
            weights._download(
                weights._HF_URL.format(
                    repo=spec.weights_repo,
                    file=f"{model_registry.TOWER_FOLDER}/{tower}.bin.zst",
                ),
                str(blob),
            )
        print("Preparing the ESM-C 6B memory-mapped weight cache...", flush=True)
        with blob.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                _write_streamed_cache(
                    params.read_records(stream), cache, "esmc", params._Q_SCALE_SUFFIX
                )
        print("ESM-C cache ready.")
    return cache
