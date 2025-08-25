
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the path to the dataset folder
DATA_PATH = "dataset"

# Load the four Excel files
df_competitive = pd.read_excel(os.path.join(DATA_PATH, "Maqsad_Competitive_Analysis_Data.xlsx"))
df_operational = pd.read_excel(os.path.join(DATA_PATH, "Maqsad_Operational_Metrics_Data.xlsx"))
df_revenue = pd.read_excel(os.path.join(DATA_PATH, "Maqsad_Revenue_and_Pricing_Data.xlsx"))
df_user_profiles = pd.read_excel(os.path.join(DATA_PATH, "Maqsad_User_Profiles_Data.xlsx"))

# Merge the dataframes based on common columns, handling potential missing columns and suffixes
# Start with df_revenue as it contains key transaction information
df_master = df_revenue.copy()

# Merge with user profiles on User_ID, using suffixes and then cleaning up duplicate user_id
df_user_profiles_cleaned = df_user_profiles.copy()
df_user_profiles_cleaned.columns = df_user_profiles_cleaned.columns.str.lower().str.replace(' ', '_')
df_master = pd.merge(df_master, df_user_profiles_cleaned, left_on='User_ID', right_on='user_id', how='left', suffixes=('_revenue', '_user'))

# Drop the redundant user_id column from the user profile dataframe after merging if it exists
if 'user_id_user' in df_master.columns:
    df_master = df_master.drop('user_id_user', axis=1)
# Rename the user_id_revenue column to user_id for consistency
df_master = df_master.rename(columns={'User_ID': 'user_id'})


# Clean column names in the merged df_master
df_master.columns = df_master.columns.str.lower().str.replace(' ', '_')

# Handle missing values
# Impute numerical columns with 0
numerical_cols_to_impute_zero = ['price_pkr', 'session_duration_min']
for col in numerical_cols_to_impute_zero:
    if col in df_master.columns:
        df_master[col] = df_master[col].fillna(0)

# Impute categorical columns with a placeholder 'Unknown'
categorical_cols_to_impute_unknown = ['revenue_model', 'product_category', 'payment_status', 'age_group', 'gender', 'region', 'educational_board', 'primary_subject', 'feature_usage']
for col in categorical_cols_to_impute_unknown:
    if col in df_master.columns:
        df_master[col] = df_master[col].fillna('Unknown')

# Drop rows where 'transaction_id', 'user_id', or 'date_revenue' are missing as these are key identifiers for the fact table
df_master.dropna(subset=['transaction_id', 'user_id', 'date'], inplace=True) # Using 'date' as it was renamed from 'date_revenue'

# Convert 'transaction_id' and 'user_id' to integer types after dropping NaNs
df_master['transaction_id'] = df_master['transaction_id'].astype(int)
df_master['user_id'] = df_master['user_id'].astype(int)

# Ensure df_master has unique column names before creating fact table
df_master = df_master.loc[:, ~df_master.columns.duplicated(keep='first')]

# Create fact table
fact_table = df_master[['date', 'transaction_id', 'price_pkr', 'session_duration_min', 'user_id', 'product_category']].copy()

# Create user dimension table from the merged df_master (which now contains user profile info)
user_dimension_table = df_master[['user_id', 'age_group', 'gender', 'region', 'educational_board', 'primary_subject', 'session_duration_min']].copy()
user_dimension_table.drop_duplicates(subset=['user_id'], inplace=True)

# Create product dimension table from the merged df_master (which now contains product info)
product_dimension_table = df_master[['product_category', 'revenue_model']].copy()
product_dimension_table.drop_duplicates(subset=['product_category'], inplace=True)

# Display head of the resulting tables (Optional in final script, but good for verification)
# print("Fact Table:")
# display(fact_table.head())

# print("\nUser Dimension Table:")
# display(user_dimension_table.head())

# print("\nProduct Dimension Table:")
# display(product_dimension_table.head())

# --- Data Mining Step ---

# Define features (X) and target variable (y)
# Use 'session_duration_min' from fact_table
numerical_features = ['session_duration_min']
target = 'price_pkr'

# Select features and target from fact_table
X = fact_table[numerical_features]
y = fact_table[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
