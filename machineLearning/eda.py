import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs("eda_results", exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("data/water_potability.csv")


# ==========================================
# 3. BASIC INFORMATION
# ==========================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== BASIC STATISTICS ==========")
print(df.describe())


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")

missing_percentage = (
    df.isnull().sum() / len(df) * 100
).sort_values(ascending=False)

print(missing_percentage)


# ==========================================
# 5. DUPLICATES
# ==========================================

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())


# ==========================================
# 6. TARGET DISTRIBUTION
# ==========================================

print("\n========== POTABILITY DISTRIBUTION ==========")
print(df["Potability"].value_counts())

print("\n========== POTABILITY PERCENTAGE ==========")
print(df["Potability"].value_counts(normalize=True) * 100)


# ==========================================
# 7. POTABILITY DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Potability"
)

plt.title("Water Potability Distribution")
plt.xlabel("Potability (0 = Not Potable, 1 = Potable)")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    "eda_results/01_potability_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 8. MISSING VALUES GRAPH
# ==========================================

missing_for_plot = missing_percentage[
    missing_percentage > 0
]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=missing_for_plot.values,
    y=missing_for_plot.index
)

plt.title("Missing Values by Feature")
plt.xlabel("Missing Values (%)")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "eda_results/02_missing_values.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 9. FEATURE DISTRIBUTIONS
# ==========================================

features = df.drop("Potability", axis=1).columns

for feature in features:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=df,
        x=feature,
        bins=30,
        kde=True
    )

    plt.title(f"{feature} Distribution")
    plt.xlabel(feature)
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.savefig(
        f"eda_results/03_distribution_{feature}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 10. CORRELATION MATRIX
# ==========================================

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "eda_results/04_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 11. CORRELATION WITH POTABILITY
# ==========================================

potability_correlation = (
    correlation["Potability"]
    .sort_values(ascending=False)
)

print("\n========== CORRELATION WITH POTABILITY ==========")
print(potability_correlation)


# ==========================================
# 12. FEATURE DISTRIBUTION BY TARGET
# ==========================================

for feature in features:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=df,
        x="Potability",
        y=feature
    )

    plt.title(
        f"{feature} Distribution by Potability"
    )

    plt.xlabel("Potability")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        f"eda_results/05_{feature}_vs_potability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 13. COMPLETION MESSAGE
# ==========================================

print("\n==========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("Graphs saved inside: eda_results/")
print("==========================================")
