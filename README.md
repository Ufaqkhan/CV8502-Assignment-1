# CV8502-Assignment 1 - Failure Analysis of Medical AI Systems (Using HAM10000)

End-to-end notebook to train/evaluate a 7-class dermoscopy classifier on **HAM10000**, run **stress tests & slices**, audit **calibration/uncertainty**, and produce **case studies** (Grad-CAM). All artifacts are saved under `output/`.

---

## Repository structure

```
.
├── Assignment-1.ipynb            # main Jupyter notebook (train → eval → stress tests → case studies)
├── requirements.txt              # pinned Python deps
├── output/
│   ├── case_studies/
│   │   ├── case-1.jpg            # original image (case 1)
│   │   ├── case-2.jpg            # original image (case 2)
│   │   └── cases.json            # machine-readable summary (GT, Pred, probs, diagnosis, mitigation)
│   ├── eval/
│   │   ├── calib_reldiag_baseline.png      # reliability diagram (uncalibrated)
│   │   ├── calib_reldiag_temp_scaling.png  # reliability diagram (temperature scaling)
│   │   ├── calib_reldiag_mc_dropout.png    # reliability diagram (MC-Dropout; optional)
│   │   └── calib_reldiag_tta.png           # reliability diagram (TTA; optional)
│   ├── COMMANDS.txt              # exact CLI commands used in the run (if executed headlessly)
│   └── history.json              # training history (loss/metrics per epoch)
└── README.md                     # this file
```

---

## Setup

```bash
# create env (Python 3.10 recommended)
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# install dependencies
pip install -r requirements.txt
```

**Dataset**
Download HAM10000 (metadata CSV + two image folders). Point the notebook to your path.

```
HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/*.jpg
└── HAM10000_images_part_2/*.jpg
```

---

## How to run

1. Open **`Assignment-1.ipynb`** in Jupyter/Lab.
2. Set `DATA_ROOT` to the HAM10000 folder.
3. Run all cells (training → clean eval → stress tests & slices → calibration → case studies).
4. Artifacts will appear in `output/` (see structure above).

> Headless option (optional): see exact commands logged in `output/COMMANDS.txt`.

---

## Repro notes (concise)

* **Model**: DenseNet-style classifier @ 224×224, cross-entropy, class-balanced sampling.
* **Training**: AdamW (lr=1e-4, wd=1e-4), standard dermoscopy augs; seed fixed (e.g., 1234).
* **Calibration/uncertainty**: temperature scaling (LBFGS) with **T** above; MC-Dropout optional.
* **Evaluation**: AUROC, AUPRC, macro-F1, Sens@95%Spec; stress tests (noise/blur/JPEG/brightness–contrast), domain shifts (color cast, down→up), slices (brightness quartiles, image size).

---

## Acknowledgements

* HAM10000: Tschandl, Rosendahl, Kittler — *Scientific Data* (2018).
* PyTorch & community libraries.
* GPT-5

---
