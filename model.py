# Import necessary libraries
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import gzip

# Load the IMDB dataset
imdb_dataset = load_dataset("imdb")['train']

# Custom function to handle negations
def handle_negations(text):
    negations = ["not", "no", "never", "n't"]
    words = text.split()
    for i in range(len(words)):
        if words[i] in negations and i+1 < len(words):
            words[i+1] = "not_" + words[i+1]
    return " ".join(words)

# Extract texts and labels from the dataset and preprocess texts
train_data = [handle_negations(item['text']) for item in imdb_dataset]
train_data_labels = [item['label'] for item in imdb_dataset]

# Initialize TfidfVectorizer with parameters
vectorizer = TfidfVectorizer(analyzer='word', max_features=10000, lowercase=True)
features = vectorizer.fit_transform(train_data).toarray()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, train_data_labels, train_size=0.75, random_state=123)

# Initialize the models
model_nb = MultinomialNB()
model_lr = LogisticRegression(max_iter=1000, n_jobs=-1)  # Increased max_iter for convergence
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123, n_jobs=-1)  # Limited max_depth for smaller model size

# Train the models
model_nb.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Predict on the validation set
y_pred_nb = model_nb.predict(X_val)
y_pred_lr = model_lr.predict(X_val)
y_pred_rf = model_rf.predict(X_val)

# Print the evaluation metrics for each model
print("Naive Bayes Accuracy:", accuracy_score(y_val, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_nb))
print()
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))
print()
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_rf))

# Function to evaluate models on test data
def evaluate_model(model, test, test_labels):
    test_features = vectorizer.transform(test).toarray()
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    confusion = confusion_matrix(test_labels, predictions)
    return accuracy, confusion

# Pickle the models and vectorizer with compression
with gzip.open('model_nb.pkl.gz', 'wb') as f:
    pickle.dump(model_nb, f)

with gzip.open('model_lr.pkl.gz', 'wb') as f:
    pickle.dump(model_lr, f)

with gzip.open('model_rf.pkl.gz', 'wb') as f:
    pickle.dump(model_rf, f)

with gzip.open('vectorizer.pkl.gz', 'wb') as f:
    pickle.dump(vectorizer, f)
