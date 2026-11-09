import pandas as pd
import math



data = {
    "ID": list(range(1, 31)),
    "Engine_Size": [
        1.2, 1.3, 1.4, 1.5, 1.6,
        1.4, 1.5, 1.6, 1.7, 1.8,
        1.3, 1.5, 1.7, 1.8, 1.9,
        2.0, 2.1, 2.2, 2.4, 2.5,
        2.6, 2.8, 3.0, 2.3, 2.5,
        2.7, 3.0, 2.2, 2.4, 2.8
    ],
    "Weight": [
        1050, 1100, 1150, 1180, 1200,
        1080, 1120, 1250, 1280, 1300,
        1070, 1190, 1230, 1350, 1380,
        1300, 1350, 1400, 1450, 1500,
        1550, 1600, 1650, 1420, 1480,
        1580, 1700, 1370, 1520, 1620
    ],
    "Class": [
        "Economy", "Economy", "Economy", "Economy", "Economy",
        "Economy", "Economy", "Economy", "Economy", "Economy",
        "Economy", "Economy", "Economy", "Economy", "Economy",
        "Sport", "Sport", "Sport", "Sport", "Sport",
        "Sport", "Sport", "Sport", "Sport", "Sport",
        "Sport", "Sport", "Sport", "Sport", "Sport"
    ]
}

df = pd.DataFrame(data)




new_engine = 2.0
new_weight = 1400

K = 5




engine_min = df["Engine_Size"].min()
engine_max = df["Engine_Size"].max()

weight_min = df["Weight"].min()
weight_max = df["Weight"].max()

df["Normalized_Engine"] = (
    (df["Engine_Size"] - engine_min)
    / (engine_max - engine_min)
)

df["Normalized_Weight"] = (
    (df["Weight"] - weight_min)
    / (weight_max - weight_min)
)



new_engine_normalized = (
    (new_engine - engine_min)
    / (engine_max - engine_min)
)

new_weight_normalized = (
    (new_weight - weight_min)
    / (weight_max - weight_min)
)




df["Distance"] = (
    (df["Normalized_Engine"] - new_engine_normalized) ** 2
    +
    (df["Normalized_Weight"] - new_weight_normalized) ** 2
).apply(math.sqrt)




df = df.sort_values("Distance")




neighbors = df.head(K)




prediction = neighbors["Class"].value_counts().idxmax()




print("\n===== KNN CAR CLASSIFICATION =====")

print("\nNew car:")
print("Engine Size:", new_engine, "L")
print("Weight:", new_weight, "kg")
print("K =", K)

print("\n5 Nearest Neighbors:")
print(
    neighbors[
        ["ID", "Engine_Size", "Weight", "Class", "Distance"]
    ].to_string(index=False)
)

print("\nClass votes:")

print(neighbors["Class"].value_counts())

print("\nPrediction:", prediction)