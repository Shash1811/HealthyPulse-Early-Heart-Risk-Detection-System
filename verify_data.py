import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/framingham.csv")

# Print basic info
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
