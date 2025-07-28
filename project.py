import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Display basic info
print("🔍 Dataset Preview:")
print(df.head())
print("\n🧼 Missing Values:")
print(df.isnull().sum())

# Drop irrelevant columns
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, inplace=True)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['alone'] = df['alone'].astype(int)

# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.show()

# Survival count by gender
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(["Male", "Female"])
plt.show()

# Age distribution by survival
sns.histplot(data=df, x='age', hue='survived', bins=20, kde=True)
plt.title("Age Distribution by Survival")
plt.show()
