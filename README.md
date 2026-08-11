# Experiment 7: Text Classification using Naïve Bayes Classifiers

## Overview
This project demonstrates text document classification using **Naïve Bayes Classifiers**. It processes and classifies text documents into distinct topic categories, comparing the performance of **Multinomial Naïve Bayes** and **Bernoulli Naïve Bayes** models using key evaluation metrics such as Accuracy and F1-Score.

---

## Aim & Objectives
* **Aim:** To implement and compare Naïve Bayes classifier variants for text document classification.
* **Objectives:**
  * Load and preprocess a multi-class text dataset.
  * Extract feature vectors using term occurrences (`CountVectorizer`).
  * Train and evaluate a **Multinomial Naïve Bayes** model (frequency-based).
  * Train and evaluate a **Bernoulli Naïve Bayes** model (binary presence/absence-based).
  * Evaluate and compare performance using Accuracy and F1-Score metrics.

---

## Categories Covered
The dataset covers 4 target topic categories:
1. `sci.space`
2. `rec.sport.baseball`
3. `comp.graphics`
4. `talk.politics.misc`

---

## Technical Stack & Dependencies
* **Language:** Python 3.x
* **Libraries Required:**
  * `scikit-learn`
  * `pandas`

---

## Project Structure
```text
├── exp7_ml.py         # Main Python execution script
├── make_zip.py        # Utility script to bundle code into a ZIP file
└── README.md          # Project documentation
