import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib # The library used for saving and loading Python objects

# --- Configuration ---
INPUT_FILENAME = 'enriched_crop_yield_preprocessed.csv'
MODEL_OUTPUT_FILENAME = 'random_forest_model.pkl'
TARGET_COLUMN = 'Yield'

# 1. Load the preprocessed data
try:
    df_preprocessed = pd.read_csv(INPUT_FILENAME)
except FileNotFoundError:
    print(f"Error: Preprocessed file '{INPUT_FILENAME}' not found. Please ensure the preprocessing step was run and the file was saved correctly.")
    exit()

# 2. Separate features (X) and target (y)
X = df_preprocessed.drop(columns=[TARGET_COLUMN])
y = df_preprocessed[TARGET_COLUMN]

# 3. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest Regressor Model
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 5. Evaluate the model (Optional, but good practice)
rf_predictions = rf_model.predict(X_test)
rf_score = r2_score(y_test, rf_predictions)
print(f"Random Forest R^2 Score on Test Data: {rf_score:.4f}\n")

# 6. Save the trained model to a .pkl file using joblib
joblib.dump(rf_model, MODEL_OUTPUT_FILENAME)

print(f"Trained Random Forest model successfully saved to: {MODEL_OUTPUT_FILENAME}")