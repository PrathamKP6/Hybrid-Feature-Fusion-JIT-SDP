# Offline Demo Guide

This guide explains how to run and verify the frozen offline Experiment 7F demo.

## 1. Open the repository

Open Git Bash and navigate to the project root:

```bash
cd /d/Hybrid-Feature-Fusion-JIT-SDP
```

Activate the project's existing virtual environment:

```bash
source .venv/Scripts/activate
```

Confirm that the environment is active and Python is available:

```bash
python --version
```

Run the following commands from the repository root with `.venv` activated.

## 2. Run the interactive CLI demo

```bash
python -m demo.cli.app
```

The CLI inspects saved validation/test predictions using the frozen Experiment 7F artifacts. It does not retrain the models or mine new commits.

Use the prompts in the terminal to:

- Inspect a commit by row number.
- Search by full or partial commit hash.
- Filter commits by project.
- Browse commits with the highest or lowest blended model-estimated risk.
- Inspect a random commit.
- Enter `q` when prompted to quit.

### Blended risk score

The frozen probability blend uses these weights:

| Model | Weight |
|---|---:|
| Random Forest | 0.15 |
| XGBoost | 0.35 |
| LightGBM | 0.50 |
| **Total** | **1.00** |

The displayed decision threshold is `0.50`. Treat the blended score as model-estimated defect risk, not as a calibrated production probability or a guarantee that a commit contains a defect.

## 3. Validate reference predictions

Run the validator as a module:

```bash
python -m demo.offline.validate_reference
```

The validator runs inference with the saved models and compares the results with the saved reference predictions. It does not retrain the models.

A successful test-split validation reports `PASS` and shows that the model and blend predictions match the reference within the configured tolerance. In the verified run, all 9,000 test commits matched within tolerance.

## 4. Run the evaluation utility

```bash
python -m demo.offline.evaluation
```

Follow the prompts printed in the terminal to use the evaluation and error-browsing features. This command was verified to run successfully in the project's virtual environment.

## 5. Important files and artifacts

| Path | Purpose |
|---|---|
| `demo/cli/app.py` | Interactive terminal demo |
| `demo/inference/predictor.py` | Loads the saved models and prepares predictions |
| `demo/offline/validate_reference.py` | Validates inference against reference predictions |
| `demo/offline/evaluation.py` | Offline evaluation metrics and error browsing |
| `results/experiment_7f_probability_blending/models/random_forest.joblib` | Frozen Random Forest model |
| `results/experiment_7f_probability_blending/models/xgboost.joblib` | Frozen XGBoost model |
| `results/experiment_7f_probability_blending/models/lightgbm.joblib` | Frozen LightGBM model |
| `results/experiment_7f_probability_blending/validation_predictions.csv` | Saved validation predictions |
| `results/experiment_7f_probability_blending/test_predictions.csv` | Saved test predictions |
| `results/pca/pca_384_model.npz` | Saved PCA artifact |

Keep the frozen models, PCA artifact, and reference prediction files unchanged when running the demo.

## 6. Troubleshooting

### `ModuleNotFoundError: No module named 'pandas'`

Make sure the project's virtual environment is activated:

```bash
source .venv/Scripts/activate
```

Then verify that Python points to the project's environment:

```bash
python -c "import sys; print(sys.executable)"
```

Use the project's existing environment rather than installing packages into a different Python installation.

### `ModuleNotFoundError: No module named 'demo'`

Run the validator and evaluation utility using module syntax from the repository root:

```bash
python -m demo.offline.validate_reference
python -m demo.offline.evaluation
```

Avoid launching the validator as `python demo/offline/validate_reference.py` in this setup, because Python may not resolve the `demo` package from that script invocation.

### A model, PCA file, or prediction CSV cannot be found

Confirm that you are running commands from the repository root and that the required artifact exists at its expected path. Do not move, rename, regenerate, or overwrite frozen artifacts just to make the demo run.

### Reference validation fails

Keep the complete output and error message. Check the active Python environment, working directory, and input files first. Do not immediately retrain models or regenerate reference predictions.

## 7. Scope and limitations

- This is an offline demo based on saved validation/test predictions, not a live service for arbitrary unseen commits.
- The demo does not retrain the models.
- The blend weights and decision threshold belong to the frozen demo configuration.
- Scores are model estimates and are not guaranteed defect labels.
