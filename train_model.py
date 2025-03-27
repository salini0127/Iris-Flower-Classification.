import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "decision_tree_model.pkl")

print("Model saved successfully!")
from sklearn.preprocessing import StandardScaler
import joblib

# Train your model (assuming X_train is your feature matrix)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the trained scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the trained model
joblib.dump(model, 'decision_tree_model.pkl')
