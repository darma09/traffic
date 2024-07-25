import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load CSV data
csv_url = 'https://raw.githubusercontent.com/darma09/traffic/main/Metro_Interstate_Traffic_Volume.csv'
data = pd.read_csv(csv_url)

# Preprocess the data - this is a generic example, adjust according to your dataset
X = data.drop(columns=['traffic_volume'])  # Replace 'traffic_volume' with the actual target column name
y = data['traffic_volume']  # Replace 'traffic_volume' with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'random_forest_model.pkl')

print("Model training complete and saved as random_forest_model.pkl")
