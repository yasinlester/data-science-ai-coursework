Data Science & AI Coursework Project
Overview

This repository contains the group project for CST3133 – Advanced Topics in Data Science and Artificial Intelligence (2024/25) at Middlesex University.

The project explores two key areas of applied data science and AI:

Structured Data Analysis (Machine Learning)

Dataset: Video Game Sales
 (cleaned Kaggle dataset)

Task: Predict global video game sales using machine learning (regression).

Model: Decision Tree Regressor (with comparisons to Linear Regression and k-NN).

Text Data Analysis (Natural Language Processing & Deep Learning)

Dataset: Amazon Product Reviews
 (public Kaggle dataset).

Task: Sentiment analysis (positive/negative classification).

Model: Feedforward Neural Network with GloVe embeddings.

Both parts follow the full data science workflow: dataset selection, preprocessing, exploratory data analysis (EDA), model development, evaluation, and ethical considerations

Installation & Setup
Requirements

The notebooks can be run locally or on Google Colab. Core libraries include:

pandas

numpy

scikit-learn

matplotlib / seaborn

nltk

tensorflow / keras

Setup

Clone the repository:

git clone https://github.com/yasinlester/data-science-ai-coursework.git
cd data-science-ai-coursework

Install dependencies (if running locally):

pip install -r requirements.txt

Part 1 – Machine Learning on Structured Data

Dataset: Video Game Sales
 (16k+ titles, attributes: Name, Platform, Year, Genre, Publisher, Regional Sales).

Task: Predict Global Sales (in millions).

Model: Decision Tree Regressor.

Key Findings:

Strongest predictors of global sales: North American and European sales.

Achieved R² = 0.86, RMSE ≈ 0.67.

Outperformed Linear Regression and k-NN.

Limitations: Dataset excludes modern digital/mobile game sales, bias towards Nintendo/NA market

AI AND DATA SCIENCE COURSEWORK …

Part 2 – Natural Language Processing & Deep Learning

Dataset: Amazon Product Reviews
 (~3M reviews, star ratings).

Task: Sentiment analysis (Positive = 4–5 stars, Negative = 1–2 stars).

Preprocessing:

Lowercasing, stopword removal, tokenisation.

Word embeddings: 100-dimensional GloVe vectors.

Model: Feedforward Neural Network.

Evaluation:

Accuracy, Precision, Recall, F1.

Confusion matrix and learning curves.

Insights: Model successfully captured sentiment polarity, though performance could be improved with advanced architectures (e.g. LSTM, BERT).

Ethical Considerations

Bias in datasets:

Video game sales skewed towards NA/EU and certain publishers (e.g. Nintendo).

Reviews dataset may reflect cultural/linguistic bias.

Mitigation:

Balanced feature selection.

Transparent reporting of dataset limitations.

Compliance: All datasets are publicly available, contain no personal data, and comply with GDPR

AI AND DATA SCIENCE COURSEWORK …

DSAI-CW-Final-V1.2 (1)

Future Work

Extend ML model with ensemble methods (Random Forest, Gradient Boosting).

Include additional features (e.g. critic scores, marketing budgets).

Use advanced NLP models (e.g. LSTM, Transformers).

Incorporate more recent datasets (digital/mobile gaming).

References
Gregorut, V. (n.d.). Video Game Sales Dataset. [online] Kaggle. Available at: 
https://www.kaggle.com/datasets/gregorut/videogamesales [Accessed 20th Jan. 2025].

Rumi, A. (n.d.) Amazon Product Reviews. Kaggle. Available at: 
https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews (Accessed: 10 March 
2025)
DSAI-CW-Final-V1.2 (1)

.
