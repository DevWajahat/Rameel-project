import pandas as pd
import os

# Define the path to the dataset folder
DATA_PATH = "dataset"

# Load the four Excel files, specifying the path to the dataset folder
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
# Ensure 'user_id' from df_user_profiles is used as the key for merging
df_master = pd.merge(df_master, df_user_profiles_cleaned, left_on='User_ID', right_on='user_id', how='left', suffixes=('_revenue', '_user'))

# Drop the redundant user_id column from the user profile dataframe after merging if it exists
if 'user_id_user' in df_master.columns:
    df_master = df_master.drop('user_id_user', axis=1)
# Rename the user_id_revenue column to user_id for consistency
df_master = df_master.rename(columns={'User_ID': 'user_id'})

# Clean column names in the merged df_master (moved before dropna)
df_master.columns = df_master.columns.str.lower().str.replace(' ', '_')

# Ensure df_master has unique column names after the initial merge steps
df_master = df_master.loc[:, ~df_master.columns.duplicated(keep='first')]

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

# Drop rows where 'transaction_id', 'user_id', or 'date' are missing as these are key identifiers for the fact table
# Use 'date' as the column name after cleaning
print("Columns in df_master before dropping:", df_master.columns) # Added print statement
df_master.dropna(subset=['transaction_id', 'user_id', 'date'], inplace=True)

# Convert 'transaction_id' and 'user_id' to integer types after dropping NaNs
df_master['transaction_id'] = df_master['transaction_id'].astype(int)
df_master['user_id'] = df_master['user_id'].astype(int)


# Create fact table
# Ensure 'session_duration_min' is included in the fact table creation
fact_table = df_master[['date', 'transaction_id', 'price_pkr', 'session_duration_min', 'user_id', 'product_category']].copy()


# Create user dimension table from the merged df_master (which now contains user profile info)
# Ensure 'session_duration_min' is included here as well for completeness in dimension table if needed
user_dimension_table = df_master[['user_id', 'age_group', 'gender', 'region', 'educational_board', 'primary_subject']].copy()
user_dimension_table.drop_duplicates(subset=['user_id'], inplace=True)

# Create product dimension table from the merged df_master (which now contains product info)
product_dimension_table = df_master[['product_category', 'revenue_model']].copy()
product_dimension_table.drop_duplicates(subset=['product_category'], inplace=True)

# Display head of the resulting tables (Optional in final script, but good for verification)
print("Fact Table:")
print(fact_table.head())

print("\nUser Dimension Table:")
print(user_dimension_table.head())

print("\nProduct Dimension Table:")
print(product_dimension_table.head())

# --- Data Mining Step (Assuming you want to keep this in this cell for a full run) ---

# Define features (X) and target variable (y)
# Use 'session_duration_min' from fact_table for prediction
numerical_features = ['session_duration_min']
target = 'price_pkr'

# Select features and target from fact_table
X = fact_table[numerical_features]
y = fact_table[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# --- Visualization Step (Assuming you want to keep this in this cell for a full run) ---
import matplotlib.pyplot as plt

# Histogram of Product Prices
plt.figure(figsize=(10, 6))
plt.hist(fact_table['price_pkr'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Product Prices')
plt.xlabel('Price (PKR)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Scatter plot of Session Duration vs Price
# Ensure 'session_duration_min' is available and correctly named for merging
# Use df_master directly as it contains both columns after processing
plt.figure(figsize=(10, 6))
plt.scatter(df_master['price_pkr'], df_master['session_duration_min'], alpha=0.5)
plt.title('Relationship between Session Duration and Price')
plt.xlabel('Price (PKR)')
plt.ylabel('Session Duration (Min)')
plt.grid(True)
plt.show()

# Bar chart of User Count by Age Group
age_group_counts = user_dimension_table['age_group'].value_counts()

plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('User Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Scatter plot of Actual vs Predicted Session Duration (Requires y_test and y_pred from model training)
# Ensure y_test and y_pred are available from the model training step
if 'y_test' in locals() and 'y_pred' in locals():
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Actual vs. Predicted Price')
    plt.xlabel('Actual Price (PKR)')
    plt.ylabel('Predicted Price (PKR)')
    plt.grid(True)
    plt.show()