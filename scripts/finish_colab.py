"""Finish NB3-NB5 after the full NB2 baseline has been saved.

Run from the repository root on the GPU runtime. Each notebook runs in a fresh
process, and writes its own log so model memory is released between stages.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.update(COMPUTE_TIER="T4", MASK_MODE="assistant-only", EPOCHS="2")
os.environ.pop("EVAL_LIMIT", None)
RESULTS = ROOT / "results"


def archive(folder: str, filename: str) -> None:
    with zipfile.ZipFile(ROOT.parent / filename, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))


def main() -> None:
    deadline = time.monotonic() + 3600
    print("Waiting for NB2's frozen full baseline and predictions", flush=True)
    while True:
        try:
            frozen = json.loads((RESULTS / "baselines_frozen.json").read_text())
            preds = json.loads((RESULTS / "baseline_predictions.json").read_text())
            assert not frozen["smoke_mode"]
            assert frozen["n_target"] == len(preds["target"]) == len(preds["preds_b"]) == 50
            assert frozen["n_regression"] == 15
            break
        except (OSError, ValueError, AssertionError, KeyError):
            if time.monotonic() > deadline:
                raise TimeoutError("NB2 did not finish: inspect results/nb2.log")
            time.sleep(15)

    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "data").glob("*.jsonl")}
    import torch
    runtime = {"gpu": torch.cuda.get_device_name(0),
               "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
               "python": sys.version,
               "packages": {n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes")},
               "input_sha256": inputs}
    (RESULTS / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2), flush=True)
    stages = ("03_train_correct", "04_misconfig_autopsy", "05_evaluate_and_verdict")
    status = {}
    try:
        for stage in stages:
            print("START", stage, flush=True)
            with (RESULTS / f"{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-u", f"notebooks/{stage}.py"],
                                        stdout=log, stderr=subprocess.STDOUT)
            status[stage] = result.returncode
            (RESULTS / "pipeline_status.json").write_text(json.dumps(status, indent=2))
            archive("results", "Lab21-Results.zip")
            if result.returncode:
                raise RuntimeError(f"{stage} failed: inspect its log")
            print("DONE", stage, flush=True)
        unchanged = all(hashlib.sha256((ROOT / "data" / n).read_bytes()).hexdigest() == h
                        for n, h in inputs.items())
        assert unchanged, "Frozen evaluation inputs changed"
        archive("adapters", "Lab21-Adapters.zip")
        print("CORE COMPLETE: Lab21-Results.zip and Lab21-Adapters.zip", flush=True)
    finally:
        archive("results", "Lab21-Results.zip")


if __name__ == "__main__":
    main()
