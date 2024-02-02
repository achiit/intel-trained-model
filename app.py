from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('text_classification_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to predict priority
def predict_priority(issue):
    # Vectorize the input issue
    issue_tfidf = tfidf_vectorizer.transform([issue])

    # Predict priority
    priority = model.predict(issue_tfidf)[0]
    return priority

@app.route('/predict_priority', methods=['POST'])
def predict_priority_api():
    try:
        # Get issue from the POST request
        data = request.get_json(force=True)
        issue = data['issue']
    
        # Get predicted priority
        predicted_priority = predict_priority(issue)

        # Return the result as JSON
        result = {'predicted_priority': predicted_priority}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

