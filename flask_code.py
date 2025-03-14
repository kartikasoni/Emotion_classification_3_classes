    
  
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import re
import json

# Load trained model and vectorizer
model_filename = "sentiment_model_logistic.pkl"  # Change this if needed
vectorizer_filename = "tfidf_vectorizer.pkl"

with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_filename, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Sentiment Labels
sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Read JSON data from the request
    data = request.get_json()

    # Get the user input text
    user_text = data.get("text", "")

    if not user_text.strip():
        return jsonify({"error": "Please enter some text!"})

    # Preprocess input text
    cleaned_text = re.sub(r"[^\w\s]", "", user_text.lower())  # Remove punctuation, lowercase
    words = cleaned_text.split()

    # Transform text using TF-IDF
    transformed_text = vectorizer.transform([user_text])

    # Predict sentiment and probabilities
    predicted_class = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]

    # Get feature names from TF-IDF
    feature_names = vectorizer.get_feature_names_out()

    # Extract TF-IDF values for words in the input
    word_tfidf_scores = {}
    feature_array = transformed_text.toarray()[0]  # Get TF-IDF values for the sentence

    for word in words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            word_tfidf_scores[word] = feature_array[idx]

    # Normalize importance scores for proper color scaling
    max_importance = max(abs(val) for val in word_tfidf_scores.values()) if word_tfidf_scores else 1
    normalized_importance = {word: (val / max_importance) for word, val in word_tfidf_scores.items()}

    # Assign color coding based on sentiment class probabilities
    highlighted_text = []
    for word in words:
        importance = normalized_importance.get(word, 0)
        opacity = max(0.3, abs(importance))  # Ensure some transparency for words with lower importance
        
        # Determine color based on sentiment
        if predicted_class == -1:  # Negative
            color = f"rgba(255, 0, 0, {opacity})"
        elif predicted_class == 1:  # Positive
            color = f"rgba(0, 128, 0, {opacity})"
        else:  # Neutral
            color = f"rgba(128, 128, 128, {opacity})"

        highlighted_text.append({"word": word, "color": color})

    # Convert probabilities to percentage and round to 2 decimal places
    probability_percentages = {
        "Negative": round(probabilities[0] * 100, 2),
        "Neutral": round(probabilities[1] * 100, 2),
        "Positive": round(probabilities[2] * 100, 2),
    }

    response = {
        "text": user_text,
        "predicted_sentiment": sentiment_labels[predicted_class],
        "probabilities": probability_percentages,  # Probabilities in percentage form
        "highlighted_text": highlighted_text
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)    