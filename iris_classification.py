# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the built-in Iris dataset
iris = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add species column (numeric)
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})  # Convert to names

# Data visualization
sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()

# Boxplots to check feature distributions
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="species", y="sepal length (cm)")
plt.title("Sepal Length Distribution")
plt.savefig("C:/Users/user/Documents/Iris-Flower-Classification/sepal_length.png")  
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="species", y="petal length (cm)")
plt.title("Petal Length Distribution")
plt.savefig("C:/Users/user/Documents/Iris-Flower-Classification/petal_length.png")  
plt.show()

# Display dataset info
print("Dataset Preview:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDataset Information:")
print(df.info())

# Encode target variable
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# Define features (X) and target (y)
X = df.drop(columns=['species'])
y = df['species']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed!")

# Train Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
dt_model.fit(X_train, y_train)

# Make Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree Model
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree Model Accuracy: {dt_accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest Model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Model Accuracy: {rf_accuracy:.2f}")

# Save models and scaler
joblib.dump(dt_model, 'decision_tree_model.pkl')  # Decision Tree Model
joblib.dump(rf_model, 'random_forest_model.pkl')  # Random Forest Model
joblib.dump(scaler, 'scaler.pkl')  # Scaler
joblib.dump(encoder, 'encoder.pkl')  # Label Encoder

print("\nModels and Scaler Saved Successfully!")

# Load the saved model for testing
loaded_model = joblib.load('decision_tree_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Test with new data (example input)
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Features of an iris flower
sample_data_scaled = loaded_scaler.transform(sample_data)  # Scale input
prediction = loaded_model.predict(sample_data_scaled)

# Decode prediction
species = ["Setosa", "Versicolor", "Virginica"]
print("Predicted Species:", species[int(prediction[0])])
