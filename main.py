#libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import pearsonr
import scipy.stats as stats

#Exercise 1 - Manual & Library Pearson Correlation

# (a) Manual Pearson Correlation Calculation
X = [791.311, 785.342, 831.433, 796.699, 728.103]
Y = [2018, 2019, 2020, 2021, 2022]

x_mean = sum(X) / len(X)
y_mean = sum(Y) / len(Y)

numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(X, Y))
denominator = (sum((x - x_mean)**2 for x in X) * sum((y - y_mean)**2 for y in Y))**0.5
r_manual = numerator / denominator

print(f"Manual Pearson r: {round(r_manual, 3)}")

# (b) Iris Dataset Feature Correlations
iris = datasets.load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

r1, p1 = pearsonr(df_iris['petal length (cm)'], df_iris['petal width (cm)'])
r2, p2 = pearsonr(df_iris['petal length (cm)'], df_iris['sepal length (cm)'])

print(f"Petal Length vs Petal Width: r = {round(r1, 3)}")
print(f"Petal Length vs Sepal Length: r = {round(r2, 3)}")

# (c) Custom Dataset Correlation Analysis
df_data = pd.read_csv("data.csv")

plt.figure(figsize=(6, 4))
plt.scatter(df_data['F1'], df_data['F2'])
plt.xlabel('F1')
plt.ylabel('F2')
plt.title('F1 vs F2 Scatterplot')
plt.show()

r_custom, p_custom = pearsonr(df_data['F1'], df_data['F2'])
print(f"F1 vs F2: r = {round(r_custom, 3)}, p-value = {p_custom}")

#Exercise 2 - Visualizations (iris dataset)

# 1. Scatterplot by Species
colors = ['red', 'green', 'blue']
plt.figure(figsize=(7, 5))
for i, species in enumerate(iris.target_names):
    mask = iris.target == i
    plt.scatter(df_iris['sepal length (cm)'][mask],
                df_iris['sepal width (cm)'][mask],
                color=colors[i], label=species)
    
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width by Species')
plt.legend()
plt.show()

# 2. Histograms & Boxplot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df_iris['petal length (cm)'].plot(kind='hist', bins=20, ax=axes[0], title='Petal Length Hist')
df_iris['sepal length (cm)'].plot(kind='hist', bins=20, ax=axes[1], title='Sepal Length Hist')
axes[2].boxplot(df_iris['petal width (cm)'])
axes[2].set_title('Petal Width Boxplot')
axes[2].set_ylabel('Petal Width (cm)')

plt.tight_layout()
plt.show()

# 3. QQ-Plot
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(df_iris['petal length (cm)'], dist="norm", plot=ax)
ax.set_title('QQ-Plot of Petal Length')
plt.show()

#Exercise 3 - Data Transformation (Min-Max Scaling)

def min_max(df, columns):
    df_new = df.copy()
    for col in columns:
        df_new[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df_new

df_transformed = min_max(df_data, ['F3'])

# Comparison Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df_data['F3'], bins=20)
axes[0].set_title('Original F3')
axes[0].set_xlabel('F3')

axes[1].hist(df_transformed['F3'], bins=20)
axes[1].set_title('Transformed F3 (own scale)')
axes[1].set_xlabel('F3 transformed')

axes[2].hist(df_transformed['F3'], bins=20)
axes[2].set_xlim(df_data['F3'].min(), df_data['F3'].max())
axes[2].set_title('Transformed F3 (original scale)')
axes[2].set_xlabel('F3 transformed (original scale)')

plt.tight_layout()
plt.show()
