# Report on emotion classification model with three classes

**Table of contents**
1. Introduction

2. Dataset choice & preprocessing steps.

3. Model selection & training details.

4. Evaluation results & analysis.

5. Web application overview.

6. Summary

**1. Introduction**

In this work, an emotion classification model has been developed using multiple Machine Learning algorithms, including Logistic Regression, Support Vector Machine (SVM), Random Forest, and Naive Bayes Classifier. The model is trained on an Amazon review dataset to classify customer reviews into three sentiment categories: Positive, Negative, and Neutral. Additionally, the model predicts the probability distribution of each class, providing insights into the likelihood of a review belonging to each sentiment category.
To enhance interpretability, the model employs a word-level highlighting mechanism:

•Negative words are displayed in red.

•Positive words are displayed in green.

•Neutral words are displayed in gray.

Furthermore, the intensity of each word’s color is dynamically adjusted based on its TF-IDF score, ensuring that words with higher importance in emotion determination are emphasized more prominently.

A detailed description of the model’s architecture, implementation, and evaluation is provided below.

**2. Dataset Choice & Preprocessing Steps**

**A. Dataset:**

The dataset used in this work is obtained from Amazon consists of text data (reviews) labeled with three distinct sentiment classes:
•Negative (-1)

•Neutral (0)

•Positive (1)

The dataset contains 20K samples, with varying numbers across sentiment categories:

•2944 instances labeled as Negative.

•1649 instances labeled as Neutral.

•15406 instances labeled as Positive.

**B. Preprocessing:**

The following preprocessing steps were applied to the dataset:

a.Text Cleaning: Punctuation marks, special characters, and numbers were removed from the text.

b.Tokenization: The text was split into words (tokens).

c.Stopwords Removal: Common words (stopwords) that do not contribute significant meaning were removed.

d.TF-IDF Vectorization: The text data was transformed into numerical form using the TF-IDF (Term Frequency-Inverse Document Frequency) approach to represent the importance of words in the text relative to the entire dataset.

**3. Model Selection & Training Details**

The whole implementation has been performed on Spyder using python 3.10.9. The following ML models were selected and trained for the task of predicting emotions based on the text input:

a.Logistic Regression (LR)

b.Support Vector Machine (SVM)

c.Random Forest Classifier (RF)

d.Naive Bayes Classifier (NB)

**4. Evaluation Results & Analysis**

•Logistic Regression and Support Vector Machine performed the best, with SVM slightly outperforming Logistic Regression in terms of overall accuracy and precision. These models were able to correctly classify the Positive sentiment with high accuracy.

•Random Forest and Naive Bayes performed worse, particularly on the Neutral class. The Random Forest model, despite having high precision on certain classes, was poor in identifying Neutral sentiments, which contributed to the reduced overall accuracy. The Naive Bayes model showed a clear bias towards Positive sentiment, which impacted its performance for Negative and Neutral classes.

•Classification Metrics:

a. Accuracy:Accuracy measures the overall correctness of the model by calculating the proportion of correctly classified instances (both positive and negative) out of the total instances.

b. Precision: Measures how many of the predicted positive instances were actually positive.

c. Recall: Measures how many of the actual positive instances were correctly predicted.

d. F1-Score: The harmonic mean of precision and recall, providing a balance between the two metrics.

<img width="861" alt="image" src="https://github.com/user-attachments/assets/02002615-a0c4-49ed-a769-26e2c68e664b" />



**5. Web Application Overview**

A Flask-based web application has been developed to deploy the trained sentiment analysis models. The user inputs text, which is processed and analyzed for sentiment. The web application performs the following tasks:

a.Preprocessing: The input text undergoes cleaning and transformation using the same preprocessing pipeline as during training.

b.Sentiment Prediction: The text is passed through the trained models to predict the sentiment (Negative, Neutral, Positive).

c.Visualization: The web application highlights words based on their importance to the sentiment prediction, using color coding:

•Red: Negative sentiment

•Green: Positive sentiment

•Gray: Neutral sentiment

d.Probability Output: The web application displays the predicted probabilities for each sentiment class in percentage form (e.g., 85% Positive, 5% Neutral, 10% Negative).

e.User Interface: The interface is simple, allowing users to input text, view highlighted words, sentiment classification, and probabilities. To run the application open the link http://127.0.0.1:5000/ on the browser and the interface is given below


<img width="442" alt="image" src="https://github.com/user-attachments/assets/38bf37b1-3408-4c74-9875-dae3ba051085" />



This web application offers a practical way to perform sentiment analysis on user-generated text and visualize the model’s behavior in real-time.


**6. Summary**
The Logistic Regression (LR) and Support Vector Machine (SVM) models demonstrated strong overall performance. Among all the models tested, SVM delivered the highest accuracy, precision, and recall, making it the most effective for emotion classification on this dataset. Due to its superior performance, SVM is the preferred model for categorizing customer reviews into positive, negative, and neutral sentiments. However, the model’s performance could be further improved by experimenting with more advanced deep learning approaches such as BERT (Bidirectional Encoder Representations from Transformers), RoBERTa (Robustly Optimized BERT Pretraining Approach), and LSTM. Although I have implemented the classification task using LSTM, this model takes significantly longer to train compared to traditional machine learning models. Additionally, exploring ensemble techniques, which combine multiple models, may further enhance the classification performance.

