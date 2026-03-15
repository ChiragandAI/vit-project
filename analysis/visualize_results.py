import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


RESULTS_FILE = "/content/drive/MyDrive/vit_experiments/results.csv"


df = pd.read_csv(RESULTS_FILE)


print("Best experiment:")

print(df.sort_values("accuracy", ascending=False).head())


# -------------------------
# Patch size vs accuracy
# -------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    x="patch",
    y="accuracy",
    data=df
)

plt.title("Patch Size vs Accuracy")

plt.savefig("patch_vs_accuracy.png")

plt.show()


# -------------------------
# Depth vs accuracy
# -------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    x="depth",
    y="accuracy",
    data=df
)

plt.title("Transformer Depth vs Accuracy")

plt.savefig("depth_vs_accuracy.png")

plt.show()


# -------------------------
# Heads vs accuracy
# -------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    x="heads",
    y="accuracy",
    data=df
)

plt.title("Attention Heads vs Accuracy")

plt.savefig("heads_vs_accuracy.png")

plt.show()


# -------------------------
# Heatmap example
# -------------------------

pivot = df.pivot_table(
    values="accuracy",
    index="patch",
    columns="depth"
)

plt.figure(figsize=(8,6))

sns.heatmap(
    pivot,
    annot=True,
    cmap="viridis"
)

plt.title("Patch vs Depth Accuracy Heatmap")

plt.show()
