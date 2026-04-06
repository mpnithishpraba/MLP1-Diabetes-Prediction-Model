import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "diabetes.csv")

df = pd.read_csv(file_path)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for c in cols:
    X[c] = X[c].replace(0, np.nan)

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df["Outcome"] = y_test
test_path = os.path.join(current_dir, "test.csv")
test_df.to_csv(test_path, index=False)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(model, X, y, cv=5)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%\n")

print("Enter the following values:")

Pregnancies = float(input("Pregnancies: "))
Glucose = float(input("Glucose: "))
BloodPressure = float(input("Blood Pressure: "))
SkinThickness = float(input("Skin Thickness: "))
Insulin = float(input("Insulin: "))
BMI = float(input("BMI: "))
DiabetesPedigreeFunction = float(input("Diabetes Pedigree Function: "))
Age = float(input("Age: "))

data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]])

for i, c in enumerate(cols):
    if data[0][i] == 0:
        data[0][i] = X[c].median()

prediction = model.predict(data)

print("\n--- RESULT ---")
if prediction[0] == 1:
    print("You may have Diabetes.")
else:
    print("You do NOT have Diabetes.")