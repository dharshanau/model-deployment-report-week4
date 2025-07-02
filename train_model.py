from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
data = load_iris()
X = data.data
y = data.target

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved!")
