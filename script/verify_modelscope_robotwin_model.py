#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "text_encoder/config.json": (855, "a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07"),
    "text_encoder/model-00001-of-00003.safetensors": (4935812536, "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58"),
    "text_encoder/model-00002-of-00003.safetensors": (4983103192, "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7"),
    "text_encoder/model-00003-of-00003.safetensors": (1442935480, "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f"),
    "text_encoder/model.safetensors.index.json": (22476, "31c4c7bcce679eaa0dd4667462394ddb013dc2f748e0bffc893dc9146a320dab"),
    "tokenizer/special_tokens_map.json": (7079, "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270"),
    "tokenizer/spiece.model": (4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    "tokenizer/tokenizer.json": (16837459, "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"),
    "tokenizer/tokenizer_config.json": (61758, "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd"),
    "transformer/config.json": (451, "814735ca2186ae717a91f19ae7d6ca3e8bd063393056e0807a881a0315a797b1"),
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": (4821987820, "405e51ec29b50f73bbf7a53bc84a4e8e5d6cdc4f0feed3e9cbf5a4e3f27cc3e1"),
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": (4821655760, "5a5ecf2bf2522152f27189709b4774fcb3c0753bd2f6cb467c4a7183361ab34a"),
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": (535373816, "12aecc30e014ae8317067c08f198e21a2c8e8d3b55712bddcdbd1f7859ab2841"),
    "transformer/diffusion_pytorch_model.safetensors.index.json": (74936, "a18d0b385025c88d29a0c1f26bbc44d86d374f48e2a0a35d5b4ce10e5e43135f"),
    "vae/config.json": (1701, "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360"),
    "vae/diffusion_pytorch_model.safetensors": (2818777808, "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    failures = []
    records = []
    for relative, (expected_size, expected_sha) in EXPECTED.items():
        path = args.model_dir / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        size = path.stat().st_size
        actual_sha = sha256(path)
        records.append({"path": relative, "size": size, "sha256": actual_sha})
        if size != expected_size:
            failures.append(f"size mismatch: {relative}: {size} != {expected_size}")
        if actual_sha != expected_sha:
            failures.append(f"sha256 mismatch: {relative}: {actual_sha} != {expected_sha}")

    config_path = args.model_dir / "transformer/config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("attn_mode") != "torch":
            failures.append(f"attn_mode must be torch, got {config.get('attn_mode')!r}")

    manifest = {
        "model": "Robbyant/lingbot-va-posttrain-robotwin",
        "modelscope_revision": "b98ce8b85cf2cddf7bee6fd23bdf38f54d39ae1c",
        "weight_revision": "201e0b2bfc30c55fb5ba6731c159c18babd10458",
        "transformer_config_revision": "05463fdc688ea303be03d6d1d1171f3ca1dc1aa1",
        "required_bytes": sum(size for size, _ in EXPECTED.values()),
        "files": records,
        "verified": not failures,
        "failures": failures,
    }
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
