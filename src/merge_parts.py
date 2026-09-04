"""merge_parts.py -- Assembles per-repeat artifacts into the frozen matrices tables read."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import ART

PART = ART / "parts"


def merge(task):
    files = sorted(PART.glob(f"{task}_r*.npz"),
                   key=lambda p: int(p.stem.split("_r")[1]))
    if not files:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    y = zs[0]["y"]
    for z in zs:
        assert np.array_equal(z["y"], y), f"{task}: label vectors differ between repeats"
    models = [k for k in zs[0].files if k not in ("y", "sex")]
    out = {m: np.vstack([z[m] for z in zs]) for m in models}
    extra = {"sex": zs[0]["sex"]} if "sex" in zs[0].files else {}
    np.savez_compressed(ART / f"oof_{task}.npz", y=y, **extra, **out)

    metas = [json.loads((PART / f"{f.stem}.json").read_text(encoding="utf-8")) for f in files]
    (ART / f"cv_{task}.json").write_text(json.dumps(dict(
        task=task, n_repeats=len(files), n=int(len(y)), d=metas[0]["d"],
        epochs=metas[0]["epochs"], meta=metas[0]["meta"],
        seconds_total=round(sum(m["seconds"] for m in metas), 1),
        repeats=[m["repeat"] for m in metas]), indent=2), encoding="utf-8")
    return len(files), len(y), len(models)


if __name__ == "__main__":
    for t in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        r = merge(t)
        print(f"{t}: {'no parts' if r is None else f'{r[0]} repeats, n={r[1]}, {r[2]} models'}")
