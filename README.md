# DS-A4 Model Comparison and Calibration

## Data and method
Use the [sklearn Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), bundled with scikit-learn 1.7.2: 569 rows, 30 numeric features, no missing values. `y = 1 - data.target` makes malignant positive. The small, complete dataset makes probability comparison easy to reproduce.

All experiment code is in `main.ipynb`. The protocol is fixed there before test evaluation: 80/20 stratified split, seed 42, identical five training folds, and one configuration per model. Logistic uses fold-local scaling and C=1; Tree uses depth 3/leaf 10; Forest uses 200 trees/depth 5/leaf 5. The budget is one configuration and five validation fits per model, not equal CPU time. Select by pooled OOF AP, with Brier as a tie-breaker.

Assess AP, ROC AUC, Brier, ten equal-width-bin ECE, and calibration curves. Paired 95% percentile intervals use 1,000 shared class-stratified test resamples; they condition on the fitted models and observed class counts. ECE is bin-dependent and Brier also measures discrimination. Probabilities are assessed as produced by the models.

Choose the threshold on training OOF predictions: minimize `(5 FN + FP)/n` subject to at most 40% expected review workload. The fixed grid is 0 to 1.01 in steps of 0.01. Ties choose fewer reviews, then a higher threshold. Sensitivity uses FN costs 1/5/10 and capacities 20%/30%/40%. These are teaching assumptions; the test workload may differ. The coarse grid's poor 20% capacity results are retained.

## Results

| Model | OOF AP | Test AP | Test ROC AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.9943 | 0.9943 | 0.9960 | 0.0213 | 0.0302 |
| Tree | 0.9313 | 0.9349 | 0.9527 | 0.0632 | 0.0533 |
| Forest | 0.9849 | 0.9916 | 0.9944 | 0.0341 | 0.0516 |

Logistic is selected. Threshold 0.30 reviews 42/114 test cases (36.8%), below the assumed 40% capacity, with one FN and one FP; cost is 6/114 = 0.0526 per case. The large-radius slice has 58 rows and 40 positives, versus 56 rows and only two positives in the small-radius slice; both errors are in the large-radius slice. Row 73 is malignant but has probability 0.0614. There is no clear Forest AP improvement: its paired difference interval includes zero.

## Reproduction and files
Use Python 3.12.14, then run inside this folder:

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (DS-A4)"
python -m nbconvert --execute --to notebook --inplace main.ipynb --ExecutePreprocessor.timeout=600
```

Alternatively restart the Jupyter kernel and run all cells. This regenerates `results.csv`; `report.pdf` is the short saved report. Both full runs completed with identical raw results.

`results.csv` keeps paired probabilities for all three models on the same 455 training OOF rows and 114 test rows, identified by `split` and original `sample_id`. It also records the chosen model, threshold, decision, failure type, and radius slice. The training median radius is 13.34. `main.ipynb` retains all model scores, intervals, paired differences, sensitivity settings, and errors. `AI_USE.md` records assistance and verification. Small samples, one cohort, sparse calibration bins, and subgroup class imbalance limit generalization; the assumed costs are not a clinical policy.

## Submission Summary

I compared three classifiers on the sklearn Breast Cancer Wisconsin dataset: logistic regression, a shallow decision tree, and a random forest. The dataset contains 569 samples and 30 numerical features, with malignant cases coded as positive. An 80/20 stratified split and five identical training folds used seed 42. Each model received one fixed configuration. Model choice used training out-of-fold average precision, and the test set was reserved for final evaluation.

Logistic regression was selected. Its test average precision was 0.9943, ROC AUC was 0.9960, Brier score was 0.0213, and ten-bin ECE was 0.0302. The random forest did not improve these point estimates. Its paired average-precision difference from logistic regression was -0.0026, with a 95% bootstrap interval from -0.0138 to 0.0062.

Using illustrative false-negative cost 5, false-positive cost 1, and 40% review capacity, training predictions selected threshold 0.30. On the test set, 42 of 114 cases were flagged, with one false negative and one false positive. Both errors occurred in the large-radius group. One malignant case received probability 0.0614, showing that strong average performance does not prevent individual failures.

Sensitivity analysis retained poor settings: a 20% capacity limit combined with the coarse threshold grid severely underused review capacity. Small test samples, sparse calibration bins, unequal subgroup class mixes, and assumed costs limit the conclusions.
