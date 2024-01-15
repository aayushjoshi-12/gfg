# Emotion Detection Model

## Overview

This project implements an emotion detection model using BERT (Bidirectional Encoder Representations from Transformers) for natural language processing. The model is trained to classify text into one of eight emotion categories. The project also includes data preprocessing, model training, and usage in a simple script.

## Emotions

The emotion categories targeted by the model are as follows:

1. Anger
2. Disgust
3. Grief
4. Joy
5. Nervousness
6. Neutral
7. Optimism
8. Sadness

## Dependencies

Make sure to install the required dependencies before running the code:

## Bash:
```
pip install tensorflow transformers scikit-learn streamlit matplotlib
```

### Additional File
```
Add Berta Folder in your local device through this link https://drive.google.com/drive/folders/10qiR2bEnyX7OY9j3K72YDgJGo8380Z88?usp=sharing
```
## Run the APP:
``` 
streamlit run app.py
```

```
Access the app in your browser at http://localhost:8501
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
  - `test.ipynb`

## How to Use
1. Enter text in the provided input field.
2. Click the "Detect Emotion" button.
3. View detailed feedback and emotion prediction results.
4. Explore the dashboard for visual representations of emotion probabilities.
