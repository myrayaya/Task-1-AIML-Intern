# Downloading the "Titanic" Dataset from Kraggle and importing it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTING DATASET AND EXPLORE BASIC INFO(NULLS, DATATYPE)

# Read the dataset
df = pd.read_csv(r'D:/all my games/programming vscode/archive/Titanic-Dataset.csv') 

# Viewing the first few rows
print(df.head())

print("\n")

# Getting information on null and datatypes
print(df.info(),"\n")
print(df.isnull().sum(), "\n")
print(df.describe(), "\n")

# HANDLING MISSING VALUES USING MEAN/MEDIAN/IMPUTATION

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# CONVERTING CATEGORICAL FEATURES TO NUMERICAL WITH ENCODING

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'])

# NORMALIZE/STANDARDIZE NUMERICAL FEATURES

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# Normalization
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
df['Fare'] = minmax.fit_transform(df[['Fare']])

# VISUALIZING OUTLIERS USING BOXPLOTS AND REMOVING THEM

# Visualizing with boxplots
sns.boxplot(df['Age'])
plt.show()

# Removing outliers
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Age'] >= lower) & (df['Age'] <= upper)]
