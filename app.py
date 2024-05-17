from flask import Flask, render_template, request
import pickle
import gzip
import torch
from transformers import BertForSequenceClassification, BertTokenizer

app = Flask(__name__)

# Load the models and vectorizer
def load_model(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

model_nb = load_model('model_nb.pkl.gz')
model_lr = load_model('model_lr.pkl.gz')
model_rf = load_model('model_rf.pkl.gz')
vectorizer = load_model('vectorizer.pkl.gz')

# Load the BERT model
with gzip.open('bert_model.pkl.gz', 'rb') as f:
    model_bert, tokenizer_bert = pickle.load(f)

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert.to(device)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Naive Bayes
@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        vectorized_text = vectorizer.transform([text])
        prediction = model_nb.predict(vectorized_text)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('naive_bayes.html', sentiment=sentiment)

# Route for Logistic Regression
@app.route('/logistic_regression', methods=['GET', 'POST'])
def logistic_regression():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        vectorized_text = vectorizer.transform([text])
        prediction = model_lr.predict(vectorized_text)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('logistic_regression.html', sentiment=sentiment)

# Route for Random Forest
@app.route('/random_forest', methods=['GET', 'POST'])
def random_forest():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        vectorized_text = vectorizer.transform([text])
        prediction = model_rf.predict(vectorized_text)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('random_forest.html', sentiment=sentiment)

# Route for BERT
@app.route('/bert', methods=['GET', 'POST'])
def bert():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model
        model_bert.eval()
        with torch.no_grad():
            outputs = model_bert(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('bert.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
