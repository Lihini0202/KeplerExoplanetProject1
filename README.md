# Exoplanet Detection with Kepler Data

Can machine learning tell a real planet from a false alarm in NASA Kepler data? This project tries two ways of answering that:

1. **XGBoost on catalogue features.** Each Kepler Object of Interest (KOI) has measured properties: orbital period, transit depth and duration, planet radius, signal-to-noise ratio, and properties of the host star.
2. **A 1D convolutional neural network on raw light curves.** It works directly from a star's brightness over time and looks for the small, regular dips caused by a planet passing in front of it.

Both models come with explanations: SHAP for XGBoost and Grad-CAM for the CNN. A Streamlit dashboard lets you explore the data, train or load the models, and classify new observations.

## Data

| File | Source | Contents |
|---|---|---|
| `datasets/cumulative.csv` | [Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) (NASA) | ~9.5k KOIs with catalogue features and a disposition: CONFIRMED, CANDIDATE or FALSE POSITIVE |
| `exoTrain.csv`, `datasets/exoTest.csv` | [Kepler Labelled Time Series Data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) | 5,657 stars, each with 3,197 flux measurements; only 42 have a confirmed planet |

`exoTrain.csv` is too large for the repo. The app downloads it from Google Drive on first run.

The light-curve data is the heart of the problem. Fewer than 1 star in 100 has a planet, so a model that always answers "no planet" is already about 99% accurate. Accuracy is the wrong yardstick here. The results below focus on recall and precision for the planet class.

## Approach

### XGBoost (catalogue features)
- The notebook keeps CONFIRMED and FALSE POSITIVE KOIs, cleans missing and infinite values, removes zero-variance columns and standardises the features.
- 70/30 stratified train/test split.
- SHAP summary plots show which features drive the predictions ([bar](shap_summary_bar.png), [dot](shap_summary_dot.png)).

### 1D-CNN (raw light curves)
- Three convolutional blocks (Conv1D → BatchNorm → MaxPool), then global average pooling, a dense layer and dropout.
- Several techniques against the class imbalance: **focal loss**, **15× oversampling** of planet stars, **class weights**, and **augmentation** (Gaussian noise plus random time shifts, so the model can't memorise where in the series a dip falls).
- **5-fold cross-validation**, with **bootstrap 95% confidence intervals** on every metric.
- **Grad-CAM** highlights which parts of a light curve the network paid attention to.

## Results

### XGBoost: CONFIRMED vs FALSE POSITIVE (test set, 2,105 KOIs)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| CONFIRMED | 0.87 | 0.88 | 0.87 |
| FALSE POSITIVE | 0.94 | 0.93 | 0.94 |

Accuracy 91.6%, ROC AUC 0.971.

### 1D-CNN: planet vs no planet (5-fold CV, about 8–9 planet stars per fold)

| Fold | Accuracy | ROC AUC (95% CI) | Planet recall | Planet precision |
|---|---|---|---|---|
| 1 | 96.6% | 0.90 (0.83–0.95) | 0.34 | 0.09 |
| 2 | 94.3% | 0.77 (0.60–0.88) | 0.00 | 0.00 |
| 3 | 96.9% | 0.41 (0.17–0.66) | 0.00 | 0.00 |
| 4 | 97.4% | 0.72 (0.49–0.89) | 0.00 | 0.00 |
| 5 | 95.7% | 0.61 (0.49–0.73) | 0.00 | 0.00 |
| **Mean** | **96.2%** | **0.68** | **0.07** | **0.02** |

## Findings

**1. High accuracy hid a CNN that almost never found a planet.** Mean accuracy was 96%, but mean recall on planet stars was 7%: in four of five folds the CNN found none. Fold 1 shows the network can pick up some signal (AUC 0.90, a third of planets found), but with only 42 planet stars, results swing heavily depending on which few land in each fold. The wide confidence intervals show it. Oversampling, focal loss and augmentation weren't enough to make up for that little data.

**2. The XGBoost score is partly NASA's own verdict.** The feature set includes `koi_score` and the four `koi_fpflag_*` columns. These come from NASA's vetting pipeline, the same process that labels a KOI a false positive. Some of XGBoost's 91.6% therefore reflects reading that verdict back, not judging from physical measurements alone. An XGBoost model trained only on physical features (period, depth, duration, radius, SNR, stellar properties) would give a fairer measure of what the features alone can do.

**3. Catalogue features are much easier to learn from than raw light curves.** XGBoost works from thousands of labelled KOIs whose most informative quantities were already extracted. The CNN has to learn transit shapes from scratch, from 42 positive examples. That gap in usable training data, more than the choice of model, explains most of the difference in results.

## Next steps

- Retrain XGBoost without `koi_score`, `koi_fpflag_*` and `rowid`, and compare the two versions.
- Preprocess light curves the standard way: detrend, normalise each star on its own, and phase-fold on the orbital period so every transit lines up.
- Train the CNN on more labelled light curves, such as the full Kepler or TESS releases, instead of 42 positives.
- Pick the CNN's decision threshold from the precision-recall curve rather than a fixed 0.5.
- Use `exoTest.csv` as a held-out test set kept apart from cross-validation.

## The dashboard

`app.py` is a Streamlit app with pages for:
- **Data exploration:** sample light curves, the disposition breakdown and feature distributions
- **Model training:** load the pre-trained models or retrain them with adjustable settings
- **Predictions:** upload a CSV of light curves (CNN) or KOI features (XGBoost) and classify it
- **Prediction analysis:** confidence and error breakdown of the latest predictions

### Run it

Requires Python 3.10 or 3.11 (for TensorFlow 2.15).

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app downloads `exoTrain.csv` into `datasets/`.

## Repository layout

```
├── app.py                                   Streamlit dashboard
├── Mini Project Source Code (Group) .ipynb  full analysis: EDA, both models, SHAP, Grad-CAM
├── train_models.py                          retrains and saves the models (written for Google Colab)
├── models/                                  pre-trained CNN, XGBoost model, scalers, label encoder
├── datasets/                                cumulative.csv and exoTest.csv
├── shap_summary_bar.png, shap_summary_dot.png
└── requirements.txt
```

This started as a group mini project.
