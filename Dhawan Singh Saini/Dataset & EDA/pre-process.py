import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- Configuration ---
INPUT_FILENAME = 'enriched_crop_yield_2.csv'
OUTPUT_FILENAME = 'enriched_crop_yield_preprocessed.csv'
TARGET_COLUMN = 'Yield'

# 1. Load the Data
df = pd.read_csv(INPUT_FILENAME)

# Separate features (X) from the target (y)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# 2. Identify Feature Types
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Define Preprocessing Pipelines
# Numerical Transformer: Scales features to have mean 0 and standard deviation 1.
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical Transformer: Converts categories into binary (0 or 1) columns.
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 4. Create a Column Transformer
# Applies the correct transformations to the correct columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# 5. Apply Preprocessing and Create Final DataFrame
X_processed_array = preprocessor.fit_transform(X)

# Get the feature names for the new DataFrame
feature_names_numerical = numerical_features
feature_names_categorical = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([feature_names_numerical, feature_names_categorical])

# Convert the processed array back to a DataFrame
X_processed_df = pd.DataFrame(
    X_processed_array,
    columns=all_feature_names
)

# 6. Combine features and target variable
y.reset_index(drop=True, inplace=True)
preprocessed_df = pd.concat([X_processed_df, y], axis=1)

# 7. Save the preprocessed data to a new CSV file
preprocessed_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"Preprocessing complete. The data has been saved to: {OUTPUT_FILENAME}")