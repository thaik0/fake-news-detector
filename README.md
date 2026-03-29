# Fake News Detector
Introductory fake news detection project in Python for me to learn Git, VS Code, and basic ML stuff.

## Project goal
I want to understand the basic supervised machine learning pipeline for text identification.

## Pipeline
- loads dataset from "news.csv"
- inspects dataset shape, columns, and sample rows through head()
- extract labels
- splits data into training and testing sets
- converts article into TF-IDF features
- trains the Passive Aggressive Classifier
- reports test accuracy, evaluate through accuracy and confusion matrix

## How to run
1. create & activate venv
2. install dependencies:
    ```bash
    pip install -r requirements.txt
3. run main.py

## Results
92.74% accuracy

## Tech stack
- Python
- pandas
- scikit-learn
- NumPy

## What I learned
- Git and GitHub workflow
- Python venv's
- supervised learning
- train/test split
- text vectorization with TD-IDF
- basic model training, specifically .fit() and .predict()
- how to evaluate models with accuracy and confusion matrix

## Next Steps
- clean up code and refactor
- improve result reporting
- compare against another classifier