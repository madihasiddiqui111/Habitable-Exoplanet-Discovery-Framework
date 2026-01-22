# Habitable-Exoplanet-Discovery-Framework
Machine learning framework for discovering potentially habitable exoplanets using astrophysical data, feature engineering, and ensemble models.

Click on the link below to watch the demo video of the web app

 https://drive.google.com/file/d/1_WKDjZJFMd12dpBovNmZk7xkcaUzsaGi/view?usp=drive_link

## Overview

The Habitable Exoplanet Discovery Framework is a data-driven machine learning project designed to identify and classify potentially habitable exoplanets using astrophysical and orbital parameters. The framework integrates advanced data preprocessing, scientifically grounded feature engineering, and ensemble machine learning models to assess planetary habitability.

The project utilizes confirmed exoplanet data from the NASA Exoplanet Archive and introduces a custom habitability evaluation approach that goes beyond traditional single-index methods by combining multiple physical and environmental indicators.

## Problem Motivation

With the growing concerns around planetary sustainability and the scientific pursuit of extraterrestrial life, identifying potentially habitable planets has become a critical research area. Although large volumes of exoplanet data are available, existing habitability models often:

* Rely on simplified assumptions

* Ignore long-term planetary stability

* Do not effectively leverage machine learning for classification

This project addresses these limitations by proposing a robust, interpretable, and scalable machine learning framework for habitability classification.

## Objectives

* Perform extensive exploratory data analysis on exoplanetary data

* Handle missing values using advanced imputation techniques

* Engineer scientifically meaningful habitability features

* Develop and compare ensemble-based classification models

* Evaluate performance using imbalance-aware metrics

## Dataset Description

**Source:** NASA Exoplanet Archive

**Dataset:** Planetary Systems Composite Data

**Total Records:** 5,599 confirmed exoplanets

**Initial Features:** 57

**Final Features Used:** 28 (after feature reduction and engineering)

**Access Date:** 12 March 2024

**Class Distribution**

* Non-Habitable

* Potentially Habitable

* Marginally Habitable

The dataset is highly imbalanced, making balanced evaluation metrics essential.

## Data Preprocessing

### Feature Reduction

* Removed 29 irrelevant or redundant features

* Dropped features with ~80% missing values

* Reduced multicollinearity and noise

### Missing Value Imputation

Imputation methods were selected based on:

* Feature distribution

* Outlier presence

* Variance and correlation impact

### Techniques Used:

* Median Imputation

* K-Nearest Neighbors (KNN) Imputation

* MICE

* XGBoost Regressor-based Imputation

## Feature Engineering
 ### Earth Similarity Index (ESI)

* Measures similarity to Earth

* Based on planetary radius and stellar flux

* Range: 0 (low similarity) → 1 (Earth-like)

 ### Atmospheric Retention (AR)

* Estimates ability to retain a stable atmosphere

* Based on escape velocity and surface temperature

* Critical for climate regulation and radiation shielding

 ### Long-Term Stability (LTS)

* Measures orbital and environmental stability

* Uses eccentricity, semi-major axis, and stellar luminosity

* Higher values indicate better habitability potential

  ## Machine Learning Models
  
### Random Forest

* Bagging-based ensemble model

* Robust to noise and missing data

* Handles non-linear relationships

### XGBoost

* Boosting-based ensemble model

* Optimized loss minimization

* Excellent performance on structured, imbalanced datasets

## Evaluation Metrics

Due to class imbalance, the following metrics were used:

* Precision

* Recall

* F1-Score

* Balanced Accuracy (primary metric)

  ## Results

| Model         | Precision  | Recall     | F1-Score   | Balanced Accuracy |
| ------------- | ---------- | ---------- | ---------- | ----------------- |
| Random Forest | 99.54%     | 99.55%     | 99.55%     | 87.77%            |
| **XGBoost**   | **99.82%** | **99.82%** | **99.81%** | **92.59%**        |

## Technology Stack

### Language

* Python 3.12

### Libraries

* NumPy
* Pandas
* Scikit-learn
* XGBoost
* SciPy
* Matplotlib
* Seaborn
* miceforest
* Streamlit

### Environment

* Google Colab / Jupyter Notebook
* Minimum 8 GB RAM

  ## Web Application (Proposed)

A Streamlit-based interactive application to:

* Predict exoplanet habitability
* Display host star characteristics
* Provide interpretable insights into predictions

  ## Future Scope

* Incorporate atmospheric absorption coefficients

* Reduce detection bias (transit & radial velocity methods)

* Extend habitability modeling across stellar classes

* Integrate Explainable AI (XAI)

* Deploy a full interactive web platform

##  Conclusion

This project demonstrates how machine learning combined with astrophysical reasoning can significantly improve exoplanet habitability assessment. The XGBoost-based framework achieves strong performance and provides a scalable foundation for future research in space analytics.

