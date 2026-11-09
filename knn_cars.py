# K-NN classification: Economy vs Sport cars
# Excel vs Python comparison with normalization

# %% 1. IMPORTS AND DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

engine = [1.2,1.3,1.4,1.5,1.6,1.4,1.5,1.6,1.7,1.8,1.3,1.5,1.7,1.8,1.9,
          2.0,2.1,2.2,2.4,2.5,2.6,2.8,3.0,2.3,2.5,2.7,3.0,2.2,2.4,2.8]
weight = [1050,1100,1150,1180,1200,1080,1120,1250,1280,1300,1070,1190,1230,1350,1380,
          1300,1350,1400,1450,1500,1550,1600,1650,1420,1480,1580,1700,1370,1520,1620]
hp     = [70,75,80,85,90,78,82,95,100,105,73,88,98,110,115,
          150,160,170,190,200,210,230,250,180,205,225,260,175,195,240]

df = pd.DataFrame({"ID": range(1, 31), "Engine": engine, "Weight": weight,
                   "HP": hp, "Class": ["Economy"] * 15 + ["Sport"] * 15})

print("=== TRAINING DATA ===")
print(df.to_string(index=False))
print("\nClass counts:")
print(df["Class"].value_counts())

features = ["Engine", "Weight", "HP"]
X = df[features].values.astype(float)
y = df["Class"].values
new = np.array([2.0, 1250, 140])          # new object, class unknown
print("\nNew object: Engine = 2.0, Weight = 1250, HP = 140, Class = ?")

# %% 2. NORMALIZATION (parameters come from the training data only)
mn, mx = X.min(axis=0), X.max(axis=0)
mean, std = X.mean(axis=0), X.std(axis=0, ddof=1)   # ddof=1 = Excel STDEV.S

datasets = {
    "Original": (X, new),
    "Min-Max":  ((X - mn) / (mx - mn), (new - mn) / (mx - mn)),
    "Z-score":  ((X - mean) / std, (new - mean) / std),
}

print("\n=== NORMALIZATION PARAMETERS ===")
print("Min :", mn)
print("Max :", mx)
print("Mean:", mean.round(4))
print("Std :", std.round(4))
print("New object, Min-Max:", datasets["Min-Max"][1].round(4))
print("New object, Z-score:", datasets["Z-score"][1].round(4))

# %% 3. DISTANCE TABLES AND 5 NEAREST NEIGHBOURS
def dist_table(Xs, xn):
    d = pd.DataFrame({"ID": df["ID"], "Class": df["Class"]})
    d["Euclidean"] = np.sqrt(((Xs - xn) ** 2).sum(axis=1))
    d["Manhattan"] = np.abs(Xs - xn).sum(axis=1)
    d["Chebyshev"] = np.abs(Xs - xn).max(axis=1)
    return d

print("\n=== 5 NEAREST NEIGHBOURS ===")
for name, (Xs, xn) in datasets.items():
    d = dist_table(Xs, xn)
    for m in ["Euclidean", "Manhattan"]:
        print(f"\n{name} - by {m}")
        print(d.sort_values(m).head(5)[["ID", "Class", m]].round(4).to_string(index=False))

# %% 4. CLASSIFICATION OF THE NEW OBJECT (K = 1, 3, 5)
rows = []
for name, (Xs, xn) in datasets.items():
    for metric in ["euclidean", "manhattan", "chebyshev"]:
        row = {"Data": name, "Metric": metric}
        for k in [1, 3, 5]:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(Xs, y)
            row[f"K={k}"] = knn.predict(xn.reshape(1, -1))[0]
        rows.append(row)

print("\n=== PREDICTIONS FOR THE NEW OBJECT ===")
print(pd.DataFrame(rows).to_string(index=False))

# %% 5. ACCURACY (leave-one-out cross-validation)
acc = []
for name, (Xs, _) in datasets.items():
    for metric in ["euclidean", "manhattan", "chebyshev"]:
        row = {"Data": name, "Metric": metric}
        for k in [1, 3, 5]:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            row[f"K={k}"] = round(cross_val_score(knn, Xs, y, cv=LeaveOneOut()).mean() * 100, 1)
        acc.append(row)

print("\n=== ACCURACY % (leave-one-out) ===")
print(pd.DataFrame(acc).to_string(index=False))

# %% 6. PLOTS
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, xcol in zip(ax, ["Weight", "Engine"]):
    for cls, c in [("Economy", "tab:blue"), ("Sport", "tab:red")]:
        m = df["Class"] == cls
        a.scatter(df.loc[m, xcol], df.loc[m, "HP"], c=c, label=cls)
    a.scatter(new[features.index(xcol)], new[2], c="green", marker="*",
              s=250, label="New object")
    a.set_xlabel(xcol)
    a.set_ylabel("HP")
    a.legend()
plt.suptitle("K-NN: cars data and new object")
plt.tight_layout()
plt.savefig("knn_plot.png", dpi=150)
plt.show()