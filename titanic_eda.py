import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")

print("First 5 rows")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop(columns=['Cabin'], inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print("\nMissing Values After Cleaning: ")
print(df.isnull().sum())


# Survival Count

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()

# Age Distribution

plt.hist(df['Age'], bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Fare Distribution

plt.hist(df['Fare'], bins=30)
plt.title("Fare Distribution")
plt.show()

# Age VS Survival

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age VS Survival")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()