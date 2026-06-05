import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data

df = load_data("data/train.csv")

print("Dataset Shape:", df.shape)
print(df.head())


target_col = "page"   

plt.figure()
df[target_col].value_counts().plot(kind="bar")
plt.title("Target Distribution")
plt.show()


df.hist(figsize=(12, 8))
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()