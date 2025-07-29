import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample synthetic dataset
data = {
    "age": np.random.randint(16, 40, 100),
    "gender": np.random.choice(["male", "female"], 100),
    "anxiety_score": np.random.randint(0, 10, 100),
    "depression_score": np.random.randint(0, 10, 100),
}

df = pd.DataFrame(data)

# Generate target: 1 = needs help, 0 = doing okay
df["label"] = ((df["anxiety_score"] + df["depression_score"]) > 10).astype(int)

# Encode gender
le = LabelEncoder()
df["gender_encoded"] = le.fit_transform(df["gender"])  # male: 0, female: 1

# Prepare training data
X = df[["age", "gender_encoded", "anxiety_score", "depression_score"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model + label encoder
with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("gender_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and encoder saved.")