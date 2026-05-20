# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()

# Step 3: Create dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Step 4: Features and labels
X = iris.data
y = iris.target

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
