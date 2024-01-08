# Emotion Detection Model

## Overview

This project implements an emotion detection model using BERT (Bidirectional Encoder Representations from Transformers) for natural language processing. The model is trained to classify text into one of eight emotion categories. The project also includes data preprocessing, model training, and usage in a simple script.

## Emotions

The emotion categories targeted by the model are as follows:

1. Admiration
2. Amusement
3. Anger
4. Annoyance
5. Approval
6. Caring
7. Confusion
8. Curiosity

## Dependencies

Make sure to install the required dependencies before running the code:

```bash
pip install tensorflow transformers scikit-learn
```


## Project Structure

- **emotions/**
  - **berta_v1/**
  - **dataset/**
  - **src/**
    - `architecture.py`
    - `preprocessing.py`
  - `readme.md`
  - `trial.ipynb`
