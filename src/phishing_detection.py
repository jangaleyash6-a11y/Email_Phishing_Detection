import pandas as pd
from preprocess import preprocess_text
from model import train_model, predict_email

# Load dataset
data = pd.read_csv("../dataset/emails.csv")

# Preprocess dataset
data["email_text"] = data["email_text"].apply(preprocess_text)

# Train model
train_model(data["email_text"], data["label"])

print("✅ Model trained successfully!")

if __name__ == "__main__":
    email = input("Enter email text: ")
    processed_email = preprocess_text(email)
    prediction = predict_email(processed_email)

    if prediction == 1:
        print("⚠️ Phishing Email Detected!")
    else:
        print("✅ Legitimate Email")
def main():
    pass
