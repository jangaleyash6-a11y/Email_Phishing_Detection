from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
model = MultinomialNB()

def train_model(emails, labels):
    X = vectorizer.fit_transform(emails)
    model.fit(X, labels)

def predict_email(email):
    X = vectorizer.transform([email])
    return model.predict(X)[0]
