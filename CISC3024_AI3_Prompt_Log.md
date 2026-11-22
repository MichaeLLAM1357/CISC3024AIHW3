# CISC3024 AI Assignment #3 — Complete prompt log

Every prompt I typed in this assignment is collected here in order. Prompts were written in Chinese, which is the language I think in; the report is in English. File attachments are shown as `[attachments]` with their paths where useful. Nothing below has been invented or reconstructed: rows marked **summary** say so because the original phrasing is not being presented as a quotation.

This file is Appendix C of `CISC3024_AI3_Report.md` in full.

---

## Stage 0 — Opening prompt (verbatim)

```
AI Assignment #3
CISC3024 Pattern Recognition
November, 2026
AI Assignment # 3 aims to find a recent PCA/LDA algorithm with applica
tion to computer vision and pattern recognition. You are expected to use an AI tool
or agent to do everything for you, including, but not limited to, algorithm searching,
programming, and even report writing.
A report in Word or PDF format should be submitted online via UMMoodle
next week and will go through a Turnitin similarity check.
Your report should contain detailed steps on how you did this assignment, includ
ing (1) how AI finds the algorithm, (2) algorithm description, (3) how AI implements
the algorithm, (4) experiment settings and results, (5) what you have learnt from
this AI assignment, and (6) a webpage link of your source codes.
Note: You are NOT allowed to write any code by yourself!  格式或風格可參考 @"C:/Users/michael/workbuddy-ai/3024HW1/outputs/CISC3024HW1/CISC3024_AI1_Report.pdf" @"D:/CISC3024/AI_HW2/GITOUTPUT/CISC3024_AI2_Report.pdf"，你先按一步步不同階段來做，算法调研与选型為第一步，有問題請提出，不要做猜測
```

Translation: find a recent PCA or LDA algorithm for computer vision and pattern recognition; the AI does the searching, programming and writing; submit a Word or PDF report through UMMoodle with the six named parts; write no code myself; follow HW1 and HW2 in format and style; work in distinct stages with the algorithm survey first; ask when there is a question instead of guessing.

Purpose: set the scope, name the binding constraint, and require staged review.

---

## Stage 1 — Cost before commitment (verbatim)

```
k=1/2/4 全跑 所需時間要多久回答我後我再決定，GitHub 倉庫LINK 後边再補，不用理會，報告請用SKILL來寫作:@skill:humanizer
```

Purpose: ask for the runtime estimate before spending GPU quota, defer the repository link, and name `humanizer` as the writing skill.

---

## Stage 2 — First configuration choice (verbatim)

```
我選A
```

Purpose: choose the 672 px / FP32 configuration as first presented. Later superseded at stage 3.

---

## Stage 3 — Switch to the paper's operating point (verbatim, abridged)

```
我重新選擇B吧， B. 672, FP16（論文工作點）	18–27 min	24–36 min	36–54 min	1.3–2.0 h，重新寫一份代碼替換原本的ipynb
```

Purpose: change to 672 px with FP16 so the result can be compared with the paper, and regenerate the notebook rather than editing it by hand.

---

## Stage 4 — Colab quota failure (verbatim, abridged)

```
@"D:/CISC3024/AI_HW3/01_subspacead_mvtec.ipynb" 運行到k=4 時配額已滿，怎辨
```

Purpose: report that the free Colab quota ran out during the k = 4 run and ask for a recovery route.

---

## Stage 5 — Port to Kaggle with checkpointing (verbatim)

```
改成KAGGLE 中運行，每完成1個K 就要SAVE 對應的數據，重新寫一次NOTEBOOOK
```

Purpose: move the experiment to Kaggle and require each finished k to be written before the next begins, so a quota cut cannot lose completed work.

---

## Stage 6 — Results handover (verbatim, abridged)

```
[attachments: k1_raw/, k2_raw/, k4_raw/, k*_summary.csv, k*_meta.json, k*_per_category.csv,
 results_main_all.csv, results_main_per_category.csv, results_main_meta.json,
 01_subspacead_mvtec.ipynb, fig_auroc_by_category.png, fig_loc_vs_k.png, plus four sample CSVs]
，01 己完成，下一步要做什么
```

Purpose: hand over the completed main grid and ask what to do next.

---

## Stage 7 — Ablation scope and caching design (summary)

The user chose Google Colab for the ablation, a backbone ablation over six representative categories (bottle, carpet, capsule, cable, screw, transistor), and proposed saving the per-patch score maps of those six categories during the first parameter combination so the remaining combinations could be scored from cached features instead of re-extracting them.

Purpose: fix the ablation budget and the single-pass design behind §4.4.

---

## Stage 8 — Ablation anchor bug (verbatim, abridged)

```
@"D:/CISC3024/AI_HW3/02_ablations.ipynb" 出現BUG
FileNotFoundError ... No such file or directory:
'/content/drive/MyDrive/CISC3024_HW3/k4_per_category.csv'
```

Purpose: report that the anchor cell assumed notebook 01 had run in the same Drive directory. Fixed by searching the Drive tree first and falling back to the values embedded from the k = 4 bundle.

---

## Stage 9 — Ablation results handover (verbatim, abridged)

```
[attachments: 02_ablations.ipynb, giant__bottle.npz, giant__capsule.npz,
 small__transistor.npz, meta_giant.json, meta_large.json, meta_small.json,
 per_category_image.csv, per_category_pixel.csv, summary.csv]
，02 己完成，結果：... 下一步要做什么
```

Purpose: hand over the completed 36-configuration ablation bundle and ask for the next step.

---

## Stage 10 — First full report draft (verbatim, abridged)

```
你是內容創作專家，你要根據上方的上下文來作一份報告，只保留給導師必須審核的部分和核心代碼，要有WORD 和PDF報告，必須使用SKILL:@skill:humanizer。LINK:https://github.com/MichaeLLAM1357/CISC3024AIHW3，REPO: D:\GITHUB\CISC3024HW3\CISC3024AIHW3
```

Purpose: draft the report in Word and PDF, keep only what the marker has to review plus the core code, and use the humanizer skill.

---

## Stage 11 — Restyle to the HW2 pattern (verbatim)

```
@"D:/CISC3024/AI_HW2/GITOUTPUT/CISC3024_AI2_Report.pdf" 請按照這個風格來做，要包含我用了哪些提示詞之類的東西，圖你要可在資料來中找，使用SKILL @skill:humanizer
```

Purpose: match the HW2 report's structure and voice, include the prompts used, build figures from the result data, and run humanizer again.

---

## What these prompts deliberately did not do

No prompt asked the AI to write code by hand into the report. Every line submitted came from the AI, as the brief requires.

No prompt asked for a target number. The choice at stage 3 was between operating points with different runtimes, not between outcomes, and the chosen point was the one the paper uses.

No prompt asked the AI to make a result look better. The follow-up to the weak k = 1 result asked what caused it, and the follow-up to the noise measurements asked for the caveat rather than a fix.

No prompt was typed after a result was seen to ask for a different result. Every later prompt was about scope, recovery or presentation.
