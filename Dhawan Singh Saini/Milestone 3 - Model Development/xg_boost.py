import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor # The model library
from sklearn.metrics import r2_score
import joblib # For saving the trained model

# --- Configuration ---
INPUT_FILENAME = 'enriched_crop_yield_preprocessed.csv'
MODEL_OUTPUT_FILENAME = 'xgboost_model.pkl'
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

# 4. Train the XGBoost Regressor Model
print("Training XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    # This objective is standard for regression problems
    objective='reg:squarederror'
)
xgb_model.fit(X_train, y_train)

# 5. Evaluate the model
xgb_predictions = xgb_model.predict(X_test)
xgb_score = r2_score(y_test, xgb_predictions)
print(f"XGBoost R^2 Score on Test Data: {xgb_score:.4f}\n")

# 6. Save the trained model to a .pkl file using joblib
joblib.dump(xgb_model, MODEL_OUTPUT_FILENAME)

print(f"Trained XGBoost model successfully saved to: {MODEL_OUTPUT_FILENAME}")