import pandas as pd
import numpy as np

# List all your Excel file names here
file_names = [
    'dataset/Maqsad_Competitive_Analysis_Data.xlsx',
    'dataset/Maqsad_Revenue_and_Pricing_Data.xlsx',
    'dataset/Maqsad_Operational_Metrics_Data.xlsx',
    'dataset/Maqsad_User_Profiles_Data.xlsx'
]

# Create an empty list to store the dataframes
all_dfs = []

# Loop through the list of files to read, clean, and transform each one
for file_name in file_names:
    try:
        # Step 1: Extract and load the data
        df = pd.read_excel(file_name)

        # Step 2: Transform the data
        
        # 2a. Standardize column names to be lowercase with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # 2b. Remove duplicate rows. This is a vital first step
        df.drop_duplicates(inplace=True)

        # 2c. Check for and convert the 'date' column to the correct datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        
        # 2d. Advanced Data Cleaning (Handling Wrong Format/Data)
        # This part of the code assumes your data has a column named 'price_pkr'
        # If your data has another currency column, change this line
        if 'price_pkr' in df.columns:
            try:
                # Remove any currency symbols or commas and convert to numeric
                df['price_pkr'] = df['price_pkr'].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df['price_pkr'] = pd.to_numeric(df['price_pkr'], errors='coerce')
            except Exception as e:
                print(f"⚠️ Warning: Could not clean 'price_pkr' in {file_name}. Error: {e}")
        
        # 2e. Fill missing values in a specific column, if needed
        # This example fills missing 'analysis_id' values with a placeholder
        if 'analysis_id' in df.columns:
            # FIX: Using the correct, modern method to avoid a FutureWarning
            df['analysis_id'] = df['analysis_id'].fillna(-999)

        # 2f. Drop rows where ALL values are missing.
        # This helps to remove completely empty rows
        df.dropna(how='all', inplace=True)
        
        # Append the cleaned and transformed dataframe to the list
        all_dfs.append(df)
        
        print(f"✅ Successfully extracted and cleaned {file_name}")

    except FileNotFoundError:
        print(f"❌ Error: The file {file_name} was not found.")
    except Exception as e:
        # Catch any other potential errors during processing
        print(f"❌ Error processing {file_name}: {e}")

# Step 3: Load the data by concatenating all dataframes
if all_dfs:
    # Concatenate all dataframes in the list into one
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Final cleaning on the combined dataset
    
    # Drop columns that are mostly empty (more than 70% missing data)
    threshold = len(master_df) * 0.3
    master_df.dropna(thresh=threshold, axis=1, inplace=True)

    print("\nAll files combined successfully!")
    print("Master DataFrame Info:")
    # Print the information for the combined dataframe
    print(master_df.info())
    print("\nFirst 5 rows of the combined data:")
    print(master_df.head())
else:
    print("\nNo dataframes were loaded. Please check the file names and paths.")

import pandas as pd
import numpy as np

# This code assumes you have already run the previous ETL script and
# have a master_df ready to go.

# =================================================================
# PART 1: CREATE THE FACT TABLE
# =================================================================
# The fact table contains all the measurable, quantitative data.
# We include user_id as a key to link back to the dimension table.
fact_table = master_df[[
    'date',
    'transaction_id',
    'price_pkr',
    'session_duration_min',
    'user_id'
]].copy()

print("✅ Fact Table Created Successfully:")
print(fact_table.info())
print("\nFirst 5 rows of the Fact Table:")
print(fact_table.head())
print("==========================================================")


# =================================================================
# PART 2: CREATE DIMENSION TABLES
# =================================================================
# Dimension tables contain descriptive attributes and must have unique rows.
# We drop duplicates to ensure a clean, normalized dimension table.

# Create the User Dimension Table
# It contains all descriptive information about the users.
user_dimension_table = master_df[[
    'user_id',
    'age_group',
    'gender',
    'region',
    'educational_board',
    'primary_subject'
]].drop_duplicates().copy()

print("\n✅ User Dimension Table Created Successfully:")
print(user_dimension_table.info())
print("\nFirst 5 rows of the User Dimension Table:")
print(user_dimension_table.head())
print("==========================================================")

# Create the Product Dimension Table
# Contains descriptive information about products and revenue models.
product_dimension_table = master_df[[
    'product_category',
    'revenue_model'
]].drop_duplicates().copy()

print("\n✅ Product Dimension Table Created Successfully:")
print(product_dimension_table.info())
print("\nProduct Dimension Table:")
print(product_dimension_table)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# This code assumes you have already run the ETL script and have the fact_table and dimension tables ready.
# We will use the fact_table, user_dimension_table, and product_dimension_table created in the previous step.

# =================================================================
# STEP 1: PREPARE THE DATA
# =================================================================

# Merge the tables to get a single dataset for modeling
# We will merge the fact table with the dimension tables on the shared keys
df_merged = pd.merge(fact_table, user_dimension_table, on='user_id', how='left')
df_merged = pd.merge(df_merged, product_dimension_table, on='product_category', how='left')

# Drop columns that are not useful for prediction
# 'transaction_id' is an identifier and not a feature
df_merged.drop(columns=['transaction_id'], inplace=True)

# Handle missing values that might affect the model.
# For numerical features, we'll fill missing values with the median.
numerical_cols = ['session_duration_min', 'price_pkr']
for col in numerical_cols:
    df_merged[col].fillna(df_merged[col].median(), inplace=True)

# For categorical features, we'll fill missing values with a new category 'unknown'
categorical_cols = ['age_group', 'gender', 'region', 'educational_board', 'primary_subject', 'revenue_model']
for col in categorical_cols:
    df_merged[col].fillna('unknown', inplace=True)

# Convert categorical features into numerical format using LabelEncoder
# This is necessary for the machine learning model to process the data
le = LabelEncoder()
for col in categorical_cols:
    df_merged[col] = le.fit_transform(df_merged[col])

# =================================================================
# STEP 2: BUILD AND TRAIN THE MODEL
# =================================================================

# Define the features (X) and the target variable (y)
# X contains all the columns we'll use to predict session duration
X = df_merged.drop(columns=['session_duration_min', 'date', 'user_id', 'price_pkr'])
# y is the target variable we want to predict
y = df_merged['session_duration_min']

# Split the data into training and testing sets
# The model will learn from the training data and be evaluated on the testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
# n_estimators is the number of trees in the forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =================================================================
# STEP 3: EVALUATE THE MODEL
# =================================================================

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==========================================================")
print("✅ Regression Model Trained and Evaluated Successfully!")
print("==========================================================")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("\n")

