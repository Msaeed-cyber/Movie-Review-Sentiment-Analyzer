import nltk
import random
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Download dataset (if not already downloaded)
nltk.download('movie_reviews')

# Step 1: Load and shuffle the dataset
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Step 2: Convert to DataFrame
df = pd.DataFrame(documents, columns=['review', 'sentiment'])

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Step 4: Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy on test set: {accuracy * 100:.2f}%\n")

# Optional: Detailed classification report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Step 7: Sample predictions
sample_reviews = [
    "The movie was absolutely wonderful. I loved every second of it!",
    "What a boring and pointless film. I regret watching it.",
    "A beautiful storyline with strong performances. Highly recommended!",
    "Terrible plot and bad acting. A complete waste of time.",
    "I was deeply moved by the characters. It was brilliant!",
    "The movie lacked direction and was extremely slow-paced.",
    "One of the best movies I’ve seen this year. Truly amazing!",
    "I couldn't finish the movie. It was just too bad.",
    "The cinematography and music were stunning. A must-watch!",
    "Awful experience. The dialogue was cringeworthy.",
    "An inspiring and heartwarming film. Loved it!",
    "Disappointing from start to finish. Nothing made sense.",
    "Fantastic acting and a gripping story.",
    "So dull and unoriginal. I almost fell asleep.",
    "It was a rollercoaster of emotions. Beautifully made.",
    "I want my time back. That was painful to sit through.",
    "Truly a masterpiece. Exceptional in every way.",
    "Worst movie of the decade. Don’t waste your money.",
    "A delightful and fun experience for the whole family.",
    "Plot holes everywhere. The writing was lazy and weak."
]

sample_vec = vectorizer.transform(sample_reviews)
sample_preds = model.predict(sample_vec)

print("🎯 Sample Predictions:")
for review, pred in zip(sample_reviews, sample_preds):
    print(f"Review: \"{review}\" → Prediction: {pred}")

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["pos", "neg"])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Optional: Save model and vectorizer
joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("\n💾 Model and vectorizer saved as 'sentiment_model.joblib' and 'vectorizer.joblib'")
