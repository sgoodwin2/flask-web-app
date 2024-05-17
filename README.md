# Sentiment Classifier Web Application

This project is a web application built using Python, Flask, HTML, and CSS to predict the sentiment (positive or negative) of a given sentence. The application utilizes various machine learning models including Naive Bayes, Logistic Regression, Random Forest, and a fine-tuned BERT model trained on the IMDB dataset. The models are trained using `scikit-learn` and `transformers`, and are served through a Flask web framework.

## Project Structure

- **app.py**: The main Flask application file that contains the routes and logic for loading models and processing user input.
- **tuned_bert.py**: Script for fine-tuning the BERT model on the IMDB dataset and saving the model.
- **templates/**: Directory containing HTML templates for the web application.
- **static/**: Directory containing static files such as CSS styles.
- **model_nb.pkl.gz**: Pickled Naive Bayes model.
- **model_lr.pkl.gz**: Pickled Logistic Regression model.
- **model_rf.pkl.gz**: Pickled Random Forest model.
- **vectorizer.pkl.gz**: Pickled CountVectorizer used for text vectorization.
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
pip install flask scikit-learn datasets transformers torch
