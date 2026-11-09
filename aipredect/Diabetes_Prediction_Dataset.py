# Diabetes Prediction - with a simple window (UI)
# Put this file and the CSV in the SAME folder, then run it.
# Needs: pip install pandas scikit-learn   (tkinter is already included with Python)

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# ------------------------------------------------
# 1. Import the data set
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
csv_files = list(BASE_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"No CSV file found in {BASE_DIR}")

df = pd.read_csv(csv_files[0])
df = df.drop_duplicates()

# ------------------------------------------------
# 2. Transform the nominal (text) attributes into numbers
# ------------------------------------------------
X = df.iloc[:, :-1]    # all columns except the last one
y = df.iloc[:, -1]     # last column = diabetes

gender_values = sorted(X["gender"].unique())
smoking_values = sorted(X["smoking_history"].unique())

X = pd.get_dummies(X, columns=["gender", "smoking_history"], drop_first=True)

# ------------------------------------------------
# 3. Split 80% train / 20% test
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------
# 4. Train the model
# ------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42,
                               class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)

# ------------------------------------------------
# 5. Predictions on the test set + validity of the model
# ------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%  |  Recall: {recall:.2f}%")


# ------------------------------------------------
# 6. Prediction for one new person (used by the window)
# ------------------------------------------------
def predict_person(person):
    """person = dictionary with the 8 values. Returns (0 or 1, probability)."""
    row = pd.DataFrame([person])
    row = pd.get_dummies(row, columns=["gender", "smoking_history"])
    # Same columns, same order as the training data (missing ones = 0)
    row = row.reindex(columns=X.columns, fill_value=0)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1]) * 100
    return prediction, probability


# ------------------------------------------------
# 7. The window (UI)
# ------------------------------------------------
window = tk.Tk()
window.title("Diabetes Prediction")
window.geometry("420x560")

tk.Label(window, text="Diabetes Prediction", font=("Arial", 16, "bold")).pack(pady=8)
tk.Label(window,
         text=f"Model accuracy: {accuracy:.2f}%   |   Recall: {recall:.2f}%",
         fg="gray30").pack()

form = tk.Frame(window)
form.pack(pady=10)

fields = {}   # keeps all the input widgets


def add_choice(row_number, label, key, values):
    tk.Label(form, text=label, anchor="w", width=22).grid(row=row_number, column=0, pady=4)
    box = ttk.Combobox(form, values=values, state="readonly", width=18)
    box.current(0)
    box.grid(row=row_number, column=1)
    fields[key] = box


def add_number(row_number, label, key, default):
    tk.Label(form, text=label, anchor="w", width=22).grid(row=row_number, column=0, pady=4)
    entry = tk.Entry(form, width=21)
    entry.insert(0, default)
    entry.grid(row=row_number, column=1)
    fields[key] = entry


add_choice(0, "Gender", "gender", gender_values)
add_number(1, "Age (years)", "age", "45")
add_choice(2, "Hypertension", "hypertension", ["No", "Yes"])
add_choice(3, "Heart disease", "heart_disease", ["No", "Yes"])
add_choice(4, "Smoking history", "smoking_history", smoking_values)
add_number(5, "BMI", "bmi", "27.0")
add_number(6, "HbA1c level", "HbA1c_level", "5.5")
add_number(7, "Blood glucose level", "blood_glucose_level", "120")

result_label = tk.Label(window, text="", font=("Arial", 13, "bold"), wraplength=380)


def on_predict():
    try:
        person = {
            "gender": fields["gender"].get(),
            "age": float(fields["age"].get()),
            "hypertension": 1 if fields["hypertension"].get() == "Yes" else 0,
            "heart_disease": 1 if fields["heart_disease"].get() == "Yes" else 0,
            "smoking_history": fields["smoking_history"].get(),
            "bmi": float(fields["bmi"].get()),
            "HbA1c_level": float(fields["HbA1c_level"].get()),
            "blood_glucose_level": float(fields["blood_glucose_level"].get()),
        }
    except ValueError:
        messagebox.showerror("Error", "Age, BMI, HbA1c and glucose must be numbers.")
        return

    prediction, _ = predict_person(person)
    if prediction == 1:
        result_label.config(text="Diabetic", fg="red")
    else:
        result_label.config(text="Not diabetic", fg="green")


tk.Button(window, text="Predict", font=("Arial", 12), width=15,
          command=on_predict).pack(pady=10)
result_label.pack(pady=10)

tk.Label(window, text="Educational project - not a medical diagnosis.",
         fg="gray50", font=("Arial", 9)).pack(side="bottom", pady=8)

window.mainloop()