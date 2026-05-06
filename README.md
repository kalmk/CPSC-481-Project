## AI Movie Classifier: Is It Kid-Friendly?
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/a4966556-dfa9-4e37-8b73-2764928e3dc6" />


## Overview
This project predicts whether a movie is suitable for children based on a review.

It uses machine learning models to:

- Classify reviews as Suitable or Not Suitable
- Show a confidence score
- Explain the reason behind the prediction

Uses Streamlit web UI to display these results.

## Requirements
- Python

## Dependencies
```pip install -r requirements.txt```

## How to train

```python svm_model.py```

```python model.py```

```python train_distilbert_safety.py```


## How to run

```streamlit run gui.py```


## Project Structure

1. Load and clean data  
2. Convert text into features (TF-IDF)  
3. Train models (Logistic Regression, SVM, DistilBERT)  
4. Tune decision threshold  
5. Make predictions  
6. Generate explanations  
7. Show results in the web app


## File Descriptions


### `data_loading.py`
- Loads datasets from CSV files  
- Checks required columns (`review`, `final_label`)  
- Cleans missing data  
- Adds supplemental training data if available  


### `text_features.py`
- Converts text into numerical features using TF-IDF  
- Uses both:
  - Word features (1–3 word phrases)  
  - Character features (to handle typos and variations)
 

### `training_utils.py`
- Helps evaluate model performance  
- Tests different probability thresholds  
- Selects the best threshold to reduce false positives


### `model.py`
- Trains a Logistic Regression model  
- Splits data into train/validation/test  
- Tunes threshold and evaluates performance  
- Saves model files


### `svm_model.py`
- Trains a calibrated SVM model  
- Similar process as `model.py`  
- Usually performs better than Logistic Regression  
- Main traditional ML model used


### `train_distilbert_safety.py`
- Trains a DistilBERT deep learning model  
- Uses HuggingFace Transformers  
- Fine-tunes on the dataset  
- Saves trained model and threshold


### `semantic_safety.py`
- Loads and runs the DistilBERT model  
- Predicts suitability probability  
- Identifies important phrases in the review


### `model_explanations.py`
- Explains predictions from traditional ML models  
- Finds important words/phrases influencing the result  
- Generates human-readable reasoning


### `pipeline.py`
- Main prediction pipeline  
- Uses DistilBERT if available, otherwise SVM  
- Returns:
  - Label  
  - Confidence score  
  - Explanation  
  - Model source
 

### `gui.py`
- Streamlit web interface  
- Lets user input a movie review  
- Displays prediction, confidence, and explanation


