import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("../dataset/phishing.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


























































































































email = ["You have won 10,000 dollars. Click now!"]
email_clean = [clean_text(email[0])]
email_vector = vectorizer.transform(email_clean)

prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("⚠️ This email is Phishing!")
else:
    print("✅ This email is Safe!")