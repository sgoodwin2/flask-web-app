# Import necessary libraries
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the IMDB dataset
imdb_dataset = load_dataset("imdb")['train']

# Extract texts and labels from the dataset
train_data = [item['text'] for item in imdb_dataset]
train_data_labels = [item['label'] for item in imdb_dataset]

# Initialize CountVectorizer with parameters
vectorizer = CountVectorizer(analyzer='word', max_features=10000, lowercase=True)
features = vectorizer.fit_transform(train_data).toarray()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, train_data_labels, train_size=0.75, random_state=123)

# Initialize the models
model_nb = MultinomialNB()
model_lr = LogisticRegression()
model_rf = RandomForestClassifier()

# Train the models
model_nb.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Predict on the validation set
y_pred_nb = model_nb.predict(X_val)
y_pred_lr = model_lr.predict(X_val)
y_pred_rf = model_rf.predict(X_val)

# Ensemble prediction: majority voting
y_pred_ens = [0] * len(y_pred_lr)
for j in range(len(y_pred_nb)):
    if y_pred_nb[j] + y_pred_lr[j] + y_pred_rf[j] >= 2:
        y_pred_ens[j] = 1

# Print the evaluation metrics for each model
print("Naive Bayes Accuracy:", accuracy_score(y_val, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_nb))
print()
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))
print()
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_rf))
print()
print("Ensemble Accuracy:", accuracy_score(y_val, y_pred_ens))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_ens))

# Function to evaluate ensemble on test data
def ensemble(test, test_labels):
    df_pred_nb = model_nb.predict(vectorizer.transform(test).toarray())
    df_pred_lr = model_lr.predict(vectorizer.transform(test).toarray())
    df_pred_rf = model_rf.predict(vectorizer.transform(test).toarray())

    df_pred_ens = [0] * len(df_pred_lr)
    for j in range(len(df_pred_nb)):
        if df_pred_nb[j] + df_pred_lr[j] + df_pred_rf[j] >= 2:
            df_pred_ens[j] = 1
    
    print("Ensemble Accuracy:", accuracy_score(test_labels, df_pred_ens))
    print("Confusion Matrix:\n", confusion_matrix(test_labels, df_pred_ens))

# Function to return ensemble predictions
def ensemble_list(test, test_labels):
    df_pred_nb = model_nb.predict(vectorizer.transform(test).toarray())
    df_pred_lr = model_lr.predict(vectorizer.transform(test).toarray())
    df_pred_rf = model_rf.predict(vectorizer.transform(test).toarray())

    df_pred_ens = [0] * len(df_pred_lr)
    for j in range(len(df_pred_nb)):
        if df_pred_nb[j] + df_pred_lr[j] + df_pred_rf[j] >= 2:
            df_pred_ens[j] = 1

    return df_pred_ens

# Pickle the models and vectorizer
with open('model_nb.pkl', 'wb') as f:
    pickle.dump(model_nb, f)

with open('model_lr.pkl', 'wb') as f:
    pickle.dump(model_lr, f)

with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
