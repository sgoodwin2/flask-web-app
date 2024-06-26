# Sentiment Classifier Web Application

This project is a web application built using Python, Flask, HTML, and CSS to predict the sentiment (positive or negative) of a given sentence. The application utilizes various machine learning models including Naive Bayes, Logistic Regression, Random Forest, and a fine-tuned BERT model trained on a IMDB movie review dataset. The models are trained using `scikit-learn` and `transformers`, and are served through a Flask web framework.

## Project Structure

- **app.py**: The main Flask application file that contains the routes and logic for loading models and processing user input.
- **tuned_bert.py**: Script for fine-tuning the BERT model on the IMDB dataset and saving the model.
- **templates/**: Directory containing HTML templates for the web application.
- **static/**: Directory containing static files such as CSS styles.
- **model_nb.pkl.gz**: Pickled Naive Bayes model.
- **model_lr.pkl.gz**: Pickled Logistic Regression model.
- **model_rf.pkl.gz**: Pickled Random Forest model.
- **vectorizer.pkl.gz**: Pickled TfidfVectorizer used for text vectorization.
- **bert_model.pkl.gz**: Pickled fine-tuned BERT model and tokenizer.

## Requirements

To run this project, you will need the following packages installed:

- Flask
- scikit-learn
- datasets
- transformers
- torch
- gzip
- pickle

You can install the required packages using pip:

```bash
pip install flask scikit-learn datasets transformers torch gzip pickle
```

## Running The App

Run the python file in the following order in order to create the pickled models berfore running the web application:

1. model.py

2. tuned_bert.py

3. app.py

Note that the 'tuned_bert.py' may take some time to run. Furthermore, the file may need to be adjusted based on the spec of you GPU and depending on weather or not you have a GPU on your machine.
