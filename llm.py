# Import necessary libraries
import pandas as pd
from sklearnex.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load and preprocess data
df = pd.read_csv('dataset.csv')

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['issue'], df['priority'], test_size=0.2, random_state=42)

# Step 3: Feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 5: Model evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the model
joblib.dump(model, 'text_classification_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
