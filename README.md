# CISC3024 AI Assignment #3 — Training-Free Few-Shot Anomaly Detection with a PCA Subspace

Reproduction of **SubspaceAD** (Lendering, Akdag & Bondarev, CVPR 2026, pp. 28557-28566, arXiv:2602.23013v3)
on MVTec-AD, plus three ablations the paper does not report: backbone size, retained variance, and
image-level aggregation.

- **Student:** LAM KA WAI (UC325629)
- **Course:** CISC3024 Pattern Recognition, University of Macau
- **Report:** [`CISC3024_AI3_Report.pdf`](CISC3024_AI3_Report.pdf) · [`CISC3024_AI3_Report.docx`](CISC3024_AI3_Report.docx) · [`CISC3024_AI3_Report.md`](CISC3024_AI3_Report.md)
- **Prompt log:** [`CISC3024_AI3_Prompt_Log.md`](CISC3024_AI3_Prompt_Log.md) — every prompt typed, in order, verbatim where the original wording exists

## Headline numbers

Mean over all 15 MVTec-AD categories, DINOv2-G, 30 rotations per shot, `pca_ev = 0.99`, `mtop1p`.

| k | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO |
|---|---|---|---|---|
| 1 | 0.9602 | 0.9833 | 0.9760 | 0.9459 |
| 2 | 0.9742 | 0.9895 | 0.9775 | 0.9482 |
| 4 | 0.9753 | 0.9900 | 0.9790 | 0.9511 |

The paper reports 0.971 image AUROC at 1-shot. The gap at `k = 1` is one category: capsule scores
0.6745, and removing it raises the 14-category mean to 0.9806.

Ablation result worth flagging: at `pca_ev = 1.00` the projection becomes the identity, the residual
vanishes, and image AUROC collapses from 0.9435 to 0.3786. The truncation is what does the detection.

## Layout

| Path | Contents |
|---|---|
| `CISC3024_AI3_Report.md` | report source, the single file both outputs are built from |
| `CISC3024_AI3_Report.docx` | Word build, Times New Roman 12 pt, A4 |
| `CISC3024_AI3_Report.pdf` | PDF build, 28 pages |
| `CISC3024_AI3_Prompt_Log.md` | Appendix C in full: all prompts, in order |
| `notebooks/01_subspacead_mvtec.ipynb` | main grid: 15 categories × k = 1, 2, 4 |
| `notebooks/02_ablations.ipynb` | single-pass ablation harness, three sweeps |
| `tools/build_notebooks.py` | generates both notebooks from one source |
| `tools/build_report.py` | builds the DOCX and the print HTML from the Markdown |
| `tools/make_fig_auroc.py` | rebuilds the per-category figure from the result CSVs |
| `tools/make_fig_qualitative.py` | rebuilds the k=1 overlay figure from the saved visualisations |
| `tools/validate_notebooks.py` | AST check over every notebook cell |
| `tools/validate_ablation.py` | the two correctness checks for the ablation harness |
| `results/main/` | per-category metrics and run metadata for k = 1, 2, 4 |
| `results/ablation/` | the 36-configuration ablation grid |
| `figures/` | every figure in the report; `figures/source/` holds the saved visualisations used for Figure 4.3.1 |

## Running it

Both notebooks are written for Google Colab with a free T4. Nothing runs locally.

1. Open `notebooks/01_subspacead_mvtec.ipynb` in Colab, run all. About 3.4 hours for the full grid.
2. Open `notebooks/02_ablations.ipynb`, run all. About 96 minutes for 36 configurations and 216
   category results.
3. `tools/build_report.py` needs `python-docx` and a local Chromium for the PDF step; the PDF in this
   repository was printed with `msedge --headless=new --print-to-pdf`.

The upstream implementation is [`CLendering/SubspaceAD`](https://github.com/CLendering/SubspaceAD) at
commit `ef56d5c`. Four patches are needed before it runs at the paper's operating point on a T4; they
are applied by the notebooks at run time, so the upstream tree is never edited by hand. Section 3.1 of
the report explains each one.

## Notes on the numbers

- `--no_aug_categories` defaults to `["transistor"]` upstream, and the runs did not override it, so
  transistor is fitted on 4 images while every other category is fitted on 124. Its AU-PRO of 0.6990
  follows from that alone.
- Two runs of an identical configuration differ by about 0.0015 pixel AUROC, so differences below
  roughly 0.005 are not interpreted anywhere in the report.
- The whole study rests on one seed.
