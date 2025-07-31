🎬 Movie Review Sentiment Analyzer
The Movie Review Sentiment Analyzer is a Python-based machine learning project that uses Natural Language Processing (NLP) to classify IMDb movie reviews as positive or negative. It uses NLTK for dataset handling, TF-IDF for feature extraction, and Multinomial Naive Bayes for sentiment classification.

📌 Features
Loads and processes real IMDb reviews from NLTK's movie_reviews dataset

Uses TF-IDF to vectorize text

Trains a Multinomial Naive Bayes classifier

Evaluates model performance using accuracy, classification report, and confusion matrix

Allows sentiment prediction on custom sample reviews

Saves trained model and vectorizer for reuse

 Requirements
Ensure the following Python libraries are installed:

pip install nltk scikit-learn pandas seaborn matplotlib joblib


sample Output:
✅ Accuracy on test set: 83.50%

📊 Classification Report:
              precision    recall  f1-score   support

   Negative       0.82      0.84      0.83       100
   Positive       0.85      0.83      0.84       100

    accuracy                           0.84       200
   macro avg       0.84      0.84      0.84       200
weighted avg       0.84      0.84      0.84       200

🎯 Sample Predictions:
Review: "The movie was absolutely wonderful. I loved every second of it!" → Prediction: pos
Review: "What a boring and pointless film. I regret watching it." → Prediction: neg
...


📁 Project Structure:
Edit
movie_sentiment_analyzer.py   # Main Python script
sentiment_model.joblib        # Trained model (generated after running)
vectorizer.joblib             # TF-IDF vectori
