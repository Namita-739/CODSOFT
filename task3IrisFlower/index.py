# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn (sklearn) libraries for ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#loading the dataset
df=pd.read_csv("IRIS.csv")

#display the first five rows
print("First five rows of the dataset")
print(df.head())

#checking for missing values
print(("\n Checking for missing values"))
print(df.isnull().sum())

#Dataset summary
print("\n Dataset summary")
print(df.info())

print(df["species"].value_counts())

# Pairplot for feature relationships
sns.pairplot(df, hue="species")
plt.show()

# Encoding species labels to numeric values
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Split features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




