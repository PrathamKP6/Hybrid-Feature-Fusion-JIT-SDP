# Streamlit Demo Guide --- Hybrid Feature Fusion Framework for JIT-SDP

This guide explains how to start and use the Streamlit graphical demo
for the Hybrid Feature Fusion Framework for Just-in-Time Software Defect
Prediction (JIT-SDP).

## 1. What the demo does

The demo presents saved Experiment 7F predictions from a hybrid model
combining: - **Traditional JIT metrics** describing code-change size,
complexity, and developer/change history. - **CodeBERT semantic
features** representing code changes, reduced using PCA. - **LLM
reasoning features** containing encoded reasoning categories and
confidence/margin scores.

You can browse saved commit predictions, inspect a commit, and review
evaluation results.

> **Important:** This is an offline, reference-based demo. It uses saved
> predictions and artifacts. It does not retrain the models or predict
> arbitrary new commits.

## 2. Requirements

-   Windows and the project repository.
-   Python and the project virtual environment (`.venv`).
-   Required project data, saved predictions, models, and PCA artifacts.
-   Streamlit and the project's Python dependencies.

Expected project root:

``` text
D:\Hybrid-Feature-Fusion-JIT-SDP
```

## 3. Start the demo

### Git Bash

1.  Open Git Bash.

2.  Change to the project root:

    ``` bash
    cd /d/Hybrid-Feature-Fusion-JIT-SDP
    ```

3.  Activate the virtual environment if it is not already active:

    ``` bash
    source .venv/Scripts/activate
    ```

    If the prompt already starts with `(.venv)`, skip this step.

4.  Run Streamlit:

    ``` bash
    python -m streamlit run demo/streamlit_app.py
    ```

5.  The app should open in your browser. If it does not, open the local
    URL shown in the terminal, normally `http://localhost:8501`.

Keep the terminal open while using the demo.

### Stop the demo

In the terminal running Streamlit, press **Ctrl+C**.

### If Streamlit is missing

With `.venv` active:

``` bash
python -m pip install streamlit
```

If the repository provides a dependency file, follow its instructions to
install all required dependencies.

## 4. Use the interface

The sidebar contains the page navigation and the **Dataset split**
selector.

### Overview

This page gives a high-level summary of the selected split and the three
feature groups.

-   **Commits in split:** number of commits in the selected split.
-   **Higher-risk estimates / Lower-risk estimates:** counts based on
    predicted classes.
-   **Higher-risk share:** percentage classified as higher risk.
-   **Feature groups:** traditional JIT metrics, CodeBERT semantic
    features, and LLM reasoning features.

Unless explicitly labelled as reference data, risk groups are model
predictions.

### Commit explorer

Use this page to find and inspect saved commit predictions.

1.  **Search commit hash:** enter a full or partial commit hash.
2.  **Clear hash search:** clears the hash filter.
3.  **Predicted risk group:** choose all predictions or filter by
    predicted risk group.
4.  **Choose project:** select a project to filter results. Leave it
    unselected to browse across projects.
5.  **Matching commits:** shows the number of rows matching the current
    filters. Results are ordered with newer years first, with projects
    mixed where possible.
6.  **Inspect a commit:** select a row in the table. The analysis
    section should display that commit's metadata, blended risk
    estimate, individual model outputs, commit message (if available),
    and reference label (if available).

The individual model outputs are from **Random Forest**, **XGBoost**,
and **LightGBM**. The blended estimate combines their saved
probabilities using the configured blend weights.

If selecting a table row does not update the analysis, upgrade Streamlit
in the active virtual environment and restart the app:

``` bash
python -m pip install --upgrade streamlit
```

### Model performance

This page evaluates Experiment 7F on the selected split.

-   **Accuracy:** fraction of predictions classified correctly.
-   **Precision:** fraction of commits predicted buggy that are actually
    buggy.
-   **Recall:** fraction of actual buggy commits identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **MCC:** Matthews correlation coefficient, which accounts for all
    four confusion-matrix outcomes.
-   **ROC-AUC:** ability to rank buggy commits above non-buggy commits
    across thresholds.
-   **PR-AUC (Average Precision):** summary of the precision--recall
    curve.
-   **Confusion matrix:** actual reference labels compared with
    predicted risk classes.
-   **Estimated risk-score distribution:** counts of commits in
    probability bands.
-   **Actual label distribution:** counts from reference labels.

**Remember:** actual buggy/non-buggy counts come from reference labels;
higher-risk/lower-risk counts come from model predictions. A higher-risk
prediction can be a false positive, and a lower-risk prediction can be a
false negative. The evaluation page reports these outcomes rather than
hiding them.

## 5. Understanding a risk estimate

The blended model produces an estimated defect-risk score between 0 and
1, shown as a percentage.

The default classification threshold is **0.50 (50%)**: - Score at or
above 50% → **Higher risk** - Score below 50% → **Lower risk**

The individual model scores can differ because the three algorithms
learn different patterns. The blended score combines their outputs
according to the configured weights.

These scores are model estimates, **not calibrated production
probabilities and not a guarantee** that a commit contains a defect.

## 6. Dataset split selector

Use the sidebar to choose an available split, such as **validation** or
**test**. The summaries and metrics correspond to the selected split and
its saved predictions.

When presenting results, state which split is selected. Do not describe
validation metrics as test metrics or combine results from different
splits.

## 7. Suggested faculty demonstration sequence

1.  Open **Overview** and explain the project goal and the three feature
    groups.
2.  Open **Commit explorer** and demonstrate filtering by hash, project,
    and predicted risk group.
3.  Select a commit and explain its metadata, blended risk estimate, and
    individual model outputs.
4.  Point out the reference label when available, distinguishing it from
    the model prediction.
5.  Open **Model performance** and explain the metrics and confusion
    matrix.
6.  Explain that the demo uses saved Experiment 7F predictions and that
    the evaluation includes both correct and incorrect classifications.

## 8. Troubleshooting

### The app does not start

-   Confirm the terminal is in the project root.
-   Confirm `.venv` is active.
-   Run `python -m streamlit run demo/streamlit_app.py`.
-   Read the terminal traceback to identify a missing module or
    artifact.

### The browser is blank or shows an old version

-   Wait for Streamlit to finish loading.
-   Refresh with **Ctrl+Shift+R**.
-   If code changed, stop the app with **Ctrl+C** and start it again.

### "Clear hash search" raises a session-state error

-   Confirm the latest `demo/streamlit_app.py` is saved.
-   Stop and restart Streamlit after replacing the file.
-   The clear button should update the search value through a callback
    rather than modifying a widget's session-state key after the widget
    has been created.

### Commit inspection does not update after selecting a row

-   Confirm the installed Streamlit version supports dataframe row
    selection.
-   Upgrade Streamlit in the active virtual environment using
    `python -m pip install --upgrade streamlit`.
-   Restart the app and try again.

### A model or data file is missing

The demo depends on saved Experiment 7F predictions, model files, and
dataset/feature artifacts. Restore the expected files from the
repository or backup. Do not replace missing predictions with invented
values.

## 9. Limitations to explain

-   The demo browses saved predictions for the available validation/test
    data; it is not a live service for arbitrary commits.
-   It does not retrain the models when it starts.
-   A risk score is an estimate, not a guarantee of a defect.
-   A threshold converts a score into a predicted class; it cannot
    eliminate false positives or false negatives.
-   Evaluation metrics depend on the selected split.

------------------------------------------------------------------------

**Project:** Hybrid Feature Fusion Framework for Just-in-Time Software
Defect Prediction\
**Demonstration model:** Experiment 7F --- JIT metrics + CodeBERT
semantic features + LLM reasoning features\
**Application:** Streamlit offline demo
