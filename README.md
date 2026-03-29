# Fake News Detector
Introductory fake news detection project in Python for me to learn Git, VS Code, and basic ML stuff.

## Project goal
I want to understand the basic supervised machine learning pipeline for text identification:
- load dataset
- inspect data
- separate inputs and labels
- split data to training and testing sets
- convert text into numeric features
- train and evaluate a classifier

## Progress
- loads dataset from "news.csv"
- inspects dataset shape, columns, and sample rows through head()
- splits data into training and testing sets
- converts article into TF-IDF features

## Tech stack
- Python
- pandas
- scikit-learn
- NumPy

## What I'm learning
- Git and GitHub workflow
- Python venv's
- supervised learning
- train/test split
- text vectorization with TD-IDF
- model evaluation basics

## How to run
1. create & activate venv
2. install dependencies:
    ```bash
    pip install -r requirements.txt
