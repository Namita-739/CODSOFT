import pandas as pd

# Load the CSV file
df = pd.read_csv("creditcard.csv")

# Show first 5 rows
print("📌 First 5 rows:")
print(df.head())

# Show basic info about columns and missing values
print("\n📌 Dataset Info:")
print(df.info())

# Show basic statistics
print("\n📌 Statistical Summary:")
print(df.describe())

# Show number of fraud vs non-fraud cases
print("\n📌 Class Distribution:")
print(df['Class'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Separate features and labels
X = df.drop("Class", axis=1)  # All columns except 'Class'
y = df["Class"]               # Only the 'Class' column

# Step 2: Normalize the data (important for models like Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:")
print("Fraud cases in y_train:", sum(y_train == 1))
print("Non-fraud cases in y_train:", sum(y_train == 0))

# Step 4: Apply SMOTE to balance training data
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("Fraud cases in y_train_resampled:", sum(y_train_resampled == 1))
print("Non-fraud cases in y_train_resampled:", sum(y_train_resampled == 0))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Initialize the model
model = LogisticRegression(max_iter=1000, random_state=42)

# Step 2: Train the model using the resampled (balanced) training set
model.fit(X_train_resampled, y_train_resampled)

# Step 3: Predict on the test set
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

