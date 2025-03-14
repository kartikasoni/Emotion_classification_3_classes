# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Mar 13 12:11:26 2025

# @author: dell
# """



import numpy as np
import pandas as pd
import nltk
import re
import string
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Download required nltk resources (punkt and stopwords)
nltk.download('punkt')  # Tokenization
nltk.download('stopwords')  # Stopwords for text preprocessing

class SentimentAnalysisModel:
    def __init__(self, dataset_path, max_features=2000, model_type="logistic"):
        # Initialize model parameters and vectorizer
        self.dataset_path = dataset_path  # Path to the dataset
        self.max_features = max_features  # Number of features for TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)  # TF-IDF vectorizer
        
        # Choose model type based on user input
        if model_type == "svm":
            self.model = SVC(kernel="linear", probability=True, random_state=42)  # SVM model
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest model
        elif model_type == "naive_bayes":
            self.model = MultinomialNB()  # Naive Bayes model
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)  # Logistic Regression model
        
        self.df = None  # DataFrame for storing dataset

    def load_data(self):
        # Load dataset from CSV file
        self.df = pd.read_csv(self.dataset_path)
        self.df = self.df[['Text', 'Score']].copy()  # Select relevant columns (Text and Score)

    def clean_text(self):
        # Function to remove HTML tags from text
        def remove_tags(string):
            if isinstance(string, str):
                return re.sub('<.*?>', ' ', string)  # Remove HTML tags
            return ''
        
        # Apply text cleaning function to remove HTML tags
        self.df['cleaned_text'] = self.df['Text'].apply(lambda x: remove_tags(x))

        # Clean text (remove non-alphabetic characters, convert to lowercase)
        corpus = []
        for text in self.df['cleaned_text']:
            review = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
            review = review.lower()  # Convert text to lowercase
            review = ' '.join([re.sub(r'[^\w\s]', '', word) for word in review.split()])  # Remove punctuation
            corpus.append(review)

        self.df['cleaned_text'] = corpus  # Store cleaned text
        # Remove rows with only punctuation or digits
        self.df = self.df[self.df['cleaned_text'].apply(lambda x: str(x) not in string.punctuation and not str(x).isdigit())]

    def assign_labels(self):
        # Function to assign labels based on the score
        def assign_new_score(rating):
            if int(rating) < 3:
                return -1  # Negative sentiment
            elif int(rating) == 3:
                return 0   # Neutral sentiment
            return 1       # Positive sentiment

        # Apply label assignment to the dataset
        self.df['Score'] = self.df['Score'].apply(assign_new_score)
        
        # Count the number of occurrences for each sentiment class
        sentiment_counts = self.df['Score'].value_counts()
        print("\nSentiment Class Distribution:")
        print(f"Negative: {sentiment_counts.get(-1, 0)}")  # Number of negative instances
        print(f"Neutral: {sentiment_counts.get(0, 0)}")  # Number of neutral instances
        print(f"Positive: {sentiment_counts.get(1, 0)}")  # Number of positive instances

    def prepare_data(self):
        # Convert text data into TF-IDF features
        X = self.vectorizer.fit_transform(self.df['cleaned_text'])
        y = self.df['Score']
        
        # Split the dataset into training and test sets (80% training, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Train the selected model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Evaluate model performance on the test data
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy and other metrics (precision, recall, F1-score)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

        # Print classification report
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")  # Print accuracy
        print(f"Precision: {precision:.4f}")  # Print precision
        print(f"Recall: {recall:.4f}")  # Print recall
        print(f"F1-Score: {fscore:.4f}")  # Print F1-score

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Negative', 'Neutral', 'Positive'], 
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def display_probabilities(self):
        # Display predicted probabilities for each class
        if hasattr(self.model, "predict_proba"):  # Ensure model supports probability prediction
            probs = self.model.predict_proba(self.X_test)
            df_probs = pd.DataFrame(probs, columns=['Negative', 'Neutral', 'Positive'])
            df_probs['Predicted Class'] = np.argmax(probs, axis=1) - 1  # Adjust indices (-1 for correct labeling)
            df_probs['Actual Class'] = self.y_test.values

            print("\nPredicted Probabilities for Each Sentence in Test Set:")
            print(df_probs.head())  # Display the probabilities for first 5 sentences
        else:
            print("This model does not support probability prediction.")

    def save_model(self, model_filename="sentiment_model.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
        # Save trained model and vectorizer as pickle files
        with open(model_filename, "wb") as model_file:
            pickle.dump(self.model, model_file)

        with open(vectorizer_filename, "wb") as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

        print("Model and vectorizer saved successfully!")

# Example Usage
if __name__ == "__main__":
    # Loop over different model types and train each model
    for model_type in ["logistic", "svm", "random_forest", "naive_bayes"]:
        print(f"\nTraining {model_type} model...")
        sentiment_model = SentimentAnalysisModel(dataset_path='review20k.csv', model_type=model_type)
        sentiment_model.load_data()  # Load dataset
        sentiment_model.clean_text()  # Clean text
        sentiment_model.assign_labels()  # Assign sentiment labels and print distribution
        sentiment_model.prepare_data()  # Prepare the data for training
        sentiment_model.train_model()  # Train the model
        sentiment_model.evaluate_model()  # Evaluate the model
        sentiment_model.display_probabilities()  # Display probabilities (if applicable)
        sentiment_model.save_model(model_filename=f"sentiment_model_{model_type}.pkl")  # Save the trained model









# import numpy as np
# import pandas as pd
# import nltk
# import re
# import string
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk import word_tokenize
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# # Download required nltk resources
# nltk.download('punkt')
# nltk.download('stopwords')

# class SentimentAnalysisModel:
#     def __init__(self, dataset_path, max_features=2000, model_type="logistic"):
#         self.dataset_path = dataset_path
#         self.max_features = max_features
#         self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        
#         # Model selection
#         if model_type == "svm":
#             self.model = SVC(kernel="linear", probability=True, random_state=42)
#         elif model_type == "random_forest":
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         elif model_type == "naive_bayes":
#             self.model = MultinomialNB()
#         else:
#             self.model = LogisticRegression(max_iter=1000, random_state=42)
        
#         self.df = None

#     def load_data(self):
#         self.df = pd.read_csv(self.dataset_path)
#         self.df = self.df[['Text', 'Score']].copy()

#     def clean_text(self):
#         def remove_tags(string):
#             if isinstance(string, str):
#                 return re.sub('<.*?>', ' ', string)
#             return ''

#         self.df['cleaned_text'] = self.df['Text'].apply(lambda x: remove_tags(x))

#         corpus = []
#         for text in self.df['cleaned_text']:
#             review = re.sub('[^a-zA-Z]', ' ', text)
#             review = review.lower()
#             review = ' '.join([re.sub(r'[^\w\s]', '', word) for word in review.split()])
#             corpus.append(review)

#         self.df['cleaned_text'] = corpus
#         self.df = self.df[self.df['cleaned_text'].apply(lambda x: str(x) not in string.punctuation and not str(x).isdigit())]

#     def assign_labels(self):
#         def assign_new_score(rating):
#             if int(rating) < 3:
#                 return -1  # Negative
#             elif int(rating) == 3:
#                 return 0   # Neutral
#             return 1       # Positive

#         self.df['Score'] = self.df['Score'].apply(assign_new_score)
        
#         # Count the number of occurrences for each sentiment class
#         sentiment_counts = self.df['Score'].value_counts()
#         print("\nSentiment Class Distribution:")
#         print(f"Negative: {sentiment_counts.get(-1, 0)}")
#         print(f"Neutral: {sentiment_counts.get(0, 0)}")
#         print(f"Positive: {sentiment_counts.get(1, 0)}")

#     def prepare_data(self):
#         X = self.vectorizer.fit_transform(self.df['cleaned_text'])
#         y = self.df['Score']
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     def train_model(self):
#         self.model.fit(self.X_train, self.y_train)

#     def evaluate_model(self):
#         y_pred = self.model.predict(self.X_test)
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')

#         print("Classification Report:\n", classification_report(self.y_test, y_pred))
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1-Score: {fscore:.4f}")

#         cm = confusion_matrix(self.y_test, y_pred)
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Negative', 'Neutral', 'Positive'], 
#                     yticklabels=['Negative', 'Neutral', 'Positive'])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         plt.show()

#     def display_probabilities(self):
#         if hasattr(self.model, "predict_proba"):  # Ensure model supports probability prediction
#             probs = self.model.predict_proba(self.X_test)
#             df_probs = pd.DataFrame(probs, columns=['Negative', 'Neutral', 'Positive'])
#             df_probs['Predicted Class'] = np.argmax(probs, axis=1) - 1  # Adjust indices (-1 for correct labeling)
#             df_probs['Actual Class'] = self.y_test.values

#             print("\nPredicted Probabilities for Each Sentence in Test Set:")
#             print(df_probs.head())
#         else:
#             print("This model does not support probability prediction.")

#     def save_model(self, model_filename="sentiment_model.pkl", vectorizer_filename="tfidf_vectorizer.pkl"):
#         with open(model_filename, "wb") as model_file:
#             pickle.dump(self.model, model_file)

#         with open(vectorizer_filename, "wb") as vectorizer_file:
#             pickle.dump(self.vectorizer, vectorizer_file)

#         print("Model and vectorizer saved successfully!")


# # Example Usage
# if __name__ == "__main__":
#     for model_type in ["logistic", "svm", "random_forest", "naive_bayes"]:
#         print(f"\nTraining {model_type} model...")
#         sentiment_model = SentimentAnalysisModel(dataset_path='review20k.csv', model_type=model_type)
#         sentiment_model.load_data()
#         sentiment_model.clean_text()
#         sentiment_model.assign_labels()  # This will now print the sentiment class distribution
#         sentiment_model.prepare_data()
#         sentiment_model.train_model()
#         sentiment_model.evaluate_model()
#         sentiment_model.display_probabilities()
#         sentiment_model.save_model(model_filename=f"sentiment_model_{model_type}.pkl")