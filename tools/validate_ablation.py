"""Validate the ablation harness against main.py, on CPU, at toy scale.

Two independent checks.

Part A isolates the one thing the harness actually changed: main.py lets
PCAModel pick k from an explained-variance target and then throws the rest of
the spectrum away, while the harness fits with no truncation and re-runs the
same selection afterwards. On the same feature matrix those two paths must
produce the same k and the same scores, bit for bit.

Part B is end-to-end. A synthetic MVTec-AD layout is written to disk, main.py
is run on it, then the harness is run on the same directory with the same
settings and the four metrics are compared. Augmentation is off and the batch
size is 1 on both sides so that there is no legitimate source of difference:
anything above 1e-6 is a bug.

Usage:  python tools/validate_ablation.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT / "vendor" / "SubspaceAD"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "tools"))

# main.py still calls np.trapz, removed in numpy 2.0. The notebook applies the
# same rename as a patch; here a one-line alias is enough.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import build_notebooks as B  # noqa: E402
from subspacead.core.pca import PCAModel  # noqa: E402
from subspacead.post_process.scoring import calculate_anomaly_scores  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# Part A: ev selection from a retained full eigendecomposition
# ---------------------------------------------------------------------------
def part_a():
    print("\n--- Part A: ev slicing vs ev at fit time ---")
    rng = np.random.default_rng(0)
    D, N = 64, 1500
    X_train = rng.normal(size=(N, D)).astype(np.float32)
    # Give the covariance a decaying spectrum so the ev target is meaningful.
    X_train *= np.logspace(0, -2, D).astype(np.float32)
    X_test = (rng.normal(size=(400, D)) * np.logspace(0, -2, D)).astype(np.float32)

    def gen():
        for i in range(0, N, 256):
            yield X_train[i:i + 256]

    num_batches = len(range(0, N, 256))

    # Path 1: exactly what main.py does.
    ref = {}
    for ev in (0.95, 0.99, 1.0):
        m = PCAModel(k=None, ev=ev, whiten=False)
        p = m.fit(gen, D, N, num_batches)
        ref[ev] = (int(m.k), calculate_anomaly_scores(X_test, p, "reconstruction", 0))

    # Path 2: what the harness does.
    m_full = PCAModel(k=None, ev=None, whiten=False)
    p_full = m_full.fit(gen, D, N, num_batches)

    check("unt truncated fit keeps every component",
          p_full["k"] == D, f"k={p_full['k']} of D={D}")
    check("unt truncated fit keeps every eigenvector",
          p_full["components"].shape == (D, D), str(p_full["components"].shape))

    for ev in (0.95, 0.99, 1.0):
        m = PCAModel(k=None, ev=ev, whiten=False)
        m.mu_ = m_full.mu_
        m.explained_variance_ = m_full.explained_variance_
        m._select_k_components(m_full.components_)
        p = m._build_pca_params()
        s = calculate_anomaly_scores(X_test, p, "reconstruction", 0)
        ref_k, ref_s = ref[ev]
        check(f"ev={ev} selects the same k", int(m.k) == ref_k,
              f"harness {int(m.k)} vs main {ref_k}")
        dmax = float(np.max(np.abs(s - ref_s)))
        check(f"ev={ev} scores match", dmax < 1e-9, f"max |delta| {dmax:.3e}")

    # Sanity: the target has to actually move k, or the test proves nothing.
    ks = [ref[ev][0] for ev in (0.95, 0.99, 1.0)]
    check("the ev targets give different k", len(set(ks)) == 3, str(ks))


# ---------------------------------------------------------------------------
# Part B: end-to-end against main.py
# ---------------------------------------------------------------------------
def make_dataset(root, category, n_train=6, n_good=3, n_bad=3, res=56, seed=0):
    """Write the directory layout MVTecADDataset expects."""
    rng = np.random.default_rng(seed)
    cat = pathlib.Path(root) / category
    for d in ("train/good", "test/good", "test/crack", "ground_truth/crack"):
        (cat / d).mkdir(parents=True, exist_ok=True)

    def write(path, defect):
        img = rng.integers(60, 180, size=(res, res, 3), dtype=np.uint8)
        mask = np.zeros((res, res), np.uint8)
        if defect:
            y, x = rng.integers(8, res - 20, size=2)
            img[y:y + 12, x:x + 12] = 255
            mask[y:y + 12, x:x + 12] = 255
        from PIL import Image
        Image.fromarray(img).save(path)
        return mask

    for i in range(n_train):
        write(cat / "train/good" / f"{i:03d}.png", False)
    for i in range(n_good):
        write(cat / "test/good" / f"{i:03d}.png", False)
    for i in range(n_bad):
        from PIL import Image
        m = write(cat / "test/crack" / f"{i:03d}.png", True)
        Image.fromarray(m).save(cat / "ground_truth/crack" / f"{i:03d}_mask.png")


def part_b(tmp):
    print("\n--- Part B: end-to-end against main.py ---")
    cat = "toycat"
    ds_root = tmp / "ds"
    make_dataset(ds_root, cat)

    ckpt = "facebook/dinov2-small"
    layers = "-4,-5"
    res, k_shot, aug = 56, 4, 0

    outdir = tmp / "main_run"
    argv = [
        "main.py",
        "--dataset_name", "mvtec_ad", "--dataset_path", str(ds_root),
        "--seed", "42", "--batch_size", "1",
        "--image_res", str(res), "--k_shot", str(k_shot),
        "--aug_count", str(aug), "--pca_ev", "0.99",
        "--agg_method", "mean", "--img_score_agg", "mtop1p",
        # The layer string starts with "-", so it has to be attached with "="
        # or argparse reads it as a flag.
        f"--layers={layers}", "--model_ckpt", ckpt,
        "--categories", cat,
        "--outdir", str(outdir),
    ]

    # main.py runs in-process rather than as a subprocess so the np.trapz alias
    # above is visible to it. The notebook applies the same rename as a patch
    # to the file; this keeps the vendored tree untouched.
    import runpy
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = argv
    print("running main.py ...", flush=True)
    ok = True
    try:
        os.chdir(REPO)
        runpy.run_path("main.py", run_name="__main__")
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    check("main.py ran", ok)
    if not ok:
        return

    hits = sorted(outdir.rglob("benchmark_results.csv"))
    check("main.py wrote a CSV", len(hits) == 1, str([str(h) for h in hits]))
    if not hits:
        return
    import pandas as pd
    main_df = pd.read_csv(hits[0])
    main_row = main_df[main_df["Category"] == cat].iloc[0]
    print("main.py :", {k: float(main_row[k]) for k in
                        ("Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO")})

    # Now the harness, same data, same settings.
    drive = tmp / "drive"
    drive.mkdir(parents=True, exist_ok=True)
    g = {
        "REPO": str(REPO), "DS_ROOT": str(ds_root), "DRIVE_DIR": str(drive),
        "PRECISION": "fp32",
    }
    exec(B.ABL_CONFIG, g)
    g.update({
        "RES": res, "K_SHOT": k_shot, "AUG_COUNT": aug,
        "EVS": [0.99], "AGGS": ["mtop1p"],
        "BATCH_TRAIN": 1, "BATCH_TEST": 1,
        "ABL_CATS": [cat],
        "BACKBONES": {"toy": (ckpt, layers)}, "ARMS": ["toy"],
    })
    exec(B.ABL_HARNESS, g)
    print("running harness ...", flush=True)
    g["eval_arm"]("toy", ckpt, layers, [cat], force=True)

    img = pd.read_csv(f"{drive}/ablation/per_category_image.csv")
    pix = pd.read_csv(f"{drive}/ablation/per_category_pixel.csv")
    har = img[(img.backbone == "toy") & (img.img_score_agg == "mtop1p")].iloc[0]
    hpx = pix[pix.backbone == "toy"].iloc[0]
    print("harness :", {
        "Image AUROC": float(har["Image AUROC"]), "Image AUPR": float(har["Image AUPR"]),
        "Pixel AUROC": float(hpx["Pixel AUROC"]), "AU-PRO": float(hpx["AU-PRO"]),
    })

    # main.py writes 4 decimals, so 1e-4 is the tightest comparison available.
    for col, hval in (("Image AUROC", har["Image AUROC"]),
                      ("Image AUPR", har["Image AUPR"]),
                      ("Pixel AUROC", hpx["Pixel AUROC"]),
                      ("AU-PRO", hpx["AU-PRO"])):
        d = abs(float(main_row[col]) - float(hval))
        check(f"{col} matches main.py", d <= 1e-4, f"delta {d:.2e}")


def main():
    part_a()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ablval_"))
    try:
        part_b(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("ABLATION HARNESS VALIDATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
