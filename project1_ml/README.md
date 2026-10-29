# CS-433 Project 1: Predicting coronary heart disease (MICHD)

We predict whether a BRFSS 2015 survey respondent has coronary heart disease (MICHD) from 321 survey features.
There are 328,135 people in the training set and 109,379 in the test set, and only 8.8% of them are positive.
Only the **Python standard library and NumPy** are used. Matplotlib is used only to draw the report figure.

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularised logistic) |
| `helpers.py` | Provided by the course: CSV loading and submission writing |
| `utils.py` | Data loading, cleaning and feature engineering, F1 metric and threshold, cross-validation |
| `models.py` | Baselines (Table 1 methods) and our 3 models: Newton logistic regression, neural network, boosted trees |
| `experiments.py` | Reproduces every number of the report |
| `run.py` | Reproduces our final submission |
| `report/` | LaTeX source of the report |

## How to run

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in a folder `data/`, then:

```bash
pip install numpy matplotlib   # matplotlib only for the figure
python run.py                  # final submission -> submissions/submission.csv (~15 min)
python experiments.py          # all tables of the report -> results/results.txt (~85 min)
```

The first run reads the CSVs (about 1.5 min) and caches them in `data/cache.npz`.
Every random step is seeded, so the results are reproducible.

## Method in short

1. **Cleaning.** BRFSS codes for "don't know"/"refused" (7/9, 77/99, …) become missing, and "none" codes (88, 888) become 0. Mixed-unit answers are decoded: weight in lb/kg, height in ft/cm, alcohol, fruit/vegetable and exercise frequencies. Survey-administration columns and columns more than 90% missing are dropped.
2. **Features.** Answers with at most 10 possible values are one-hot encoded. Numeric answers get log(1+x) if skewed, then median imputation and standardisation.
3. **Models.**
   - A: L2-regularised logistic regression solved with Newton's method, plus pairwise products of the 20 features most correlated with the label.
   - B: 3 small neural networks (128-64 ReLU units, Adam), averaged.
   - C: gradient-boosted decision trees (300 trees of depth 5).
4. **Prediction.** The three probabilities are averaged. A person is predicted positive if the average is at least the threshold that maximises F1 in 5-fold cross-validation.
