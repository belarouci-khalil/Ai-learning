
# Credit Card Fraud Detection - Random Forest with Tkinter UI

from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import kagglehub
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


# ------------------------------------------------
# 1. Download and import the dataset
# ------------------------------------------------

print("Downloading Credit Card Fraud Detection dataset...")

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

csv_files = list(Path(path).rglob("*.csv"))

if not csv_files:
    raise SystemExit("No CSV file found.")

df = pd.read_csv(csv_files[0])
df = df.drop_duplicates()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

feature_names = X.columns.tolist()


# ------------------------------------------------
# 3. Split 80% training / 20% testing
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# 4. Train Random Forest model
# ------------------------------------------------

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ------------------------------------------------
# 5. Evaluate model
# ------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred, zero_division=0) * 100

print("\nCREDIT CARD FRAUD DETECTION")
print("---------------------------")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Recall: {recall:.2f}%")


# ------------------------------------------------
# 6. Predict a new transaction
# ------------------------------------------------

def predict_transaction(transaction):

    row = pd.DataFrame([transaction])
    row = row.reindex(columns=X.columns, fill_value=0)

    prediction = int(model.predict(row)[0])

    fraud_index = list(model.classes_).index(1)

    probability = float(
        model.predict_proba(row)[0, fraud_index]
    ) * 100

    return prediction, probability


# ------------------------------------------------
# 7. Tkinter graphical interface
# ------------------------------------------------

window = tk.Tk()
window.title("Credit Card Fraud Detection")
window.geometry("700x750")
window.configure(bg="#f4f6f9")


# Title
tk.Label(
    window,
    text="Credit Card Fraud Detection",
    font=("Arial", 20, "bold"),
    bg="#f4f6f9",
    fg="#1f2937"
).pack(pady=12)


# Model performance
tk.Label(
    window,
    text=f"Model Accuracy: {accuracy:.2f}%   |   Recall: {recall:.2f}%",
    font=("Arial", 11),
    bg="#f4f6f9",
    fg="#374151"
).pack(pady=5)


tk.Label(
    window,
    text="Enter Transaction Information",
    font=("Arial", 12, "bold"),
    bg="#f4f6f9"
).pack(pady=10)


# ------------------------------------------------
# 8. Scrollable form
# ------------------------------------------------

container = tk.Frame(window, bg="#f4f6f9")
container.pack(fill="both", expand=True, padx=20, pady=10)

canvas = tk.Canvas(
    container,
    bg="#f4f6f9",
    highlightthickness=0
)

scrollbar = tk.Scrollbar(
    container,
    orient="vertical",
    command=canvas.yview
)

form = tk.Frame(canvas, bg="#f4f6f9")

form.bind(
    "<Configure>",
    lambda event: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window(
    (0, 0),
    window=form,
    anchor="nw"
)

canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


fields = {}


# ------------------------------------------------
# 9. Generate input fields with random values
# ------------------------------------------------

for i, feature in enumerate(feature_names):

    tk.Label(
        form,
        text=feature,
        font=("Arial", 10, "bold"),
        width=15,
        anchor="w",
        bg="#f4f6f9"
    ).grid(row=i, column=0, padx=10, pady=5)

    entry = tk.Entry(
        form,
        width=30,
        font=("Arial", 10)
    )

    # Random default values directly in the UI
    if feature == "Time":
        default = round(random.uniform(0, 172792), 2)

    elif feature == "Amount":
        default = round(random.uniform(1, 2000), 2)

    else:
        default = round(random.uniform(-3, 3), 6)

    entry.insert(0, str(default))

    entry.grid(row=i, column=1, padx=10, pady=5)

    fields[feature] = entry


# ------------------------------------------------
# 10. Prediction result
# ------------------------------------------------

result_label = tk.Label(
    window,
    text="",
    font=("Arial", 14, "bold"),
    bg="#f4f6f9",
    wraplength=600
)


def on_predict():

    try:

        transaction = {}

        for feature in feature_names:

            value = float(fields[feature].get())
            transaction[feature] = value

    except ValueError:

        messagebox.showerror(
            "Input Error",
            "Please enter valid numerical values."
        )

        return

    prediction, probability = predict_transaction(transaction)

    if prediction == 1:

        result_label.config(
            text=f"FRAUDULENT TRANSACTION\nFraud probability: {probability:.2f}%",
            fg="red"
        )

    else:

        result_label.config(
            text=f"LEGITIMATE TRANSACTION\nFraud probability: {probability:.2f}%",
            fg="green"
        )


# ------------------------------------------------
# 11. Predict button
# ------------------------------------------------

tk.Button(
    window,
    text="Predict Transaction",
    font=("Arial", 12, "bold"),
    width=22,
    bg="#2563eb",
    fg="white",
    command=on_predict
).pack(pady=12)


result_label.pack(pady=10)


# Footer
tk.Label(
    window,
    text="Educational Machine Learning Project",
    font=("Arial", 9),
    fg="gray",
    bg="#f4f6f9"
).pack(pady=8)


# ------------------------------------------------
# 12. Start application
# ------------------------------------------------

window.mainloop()
