import pandas as pd
import os

# List of CSV files to combine
csv_files = [
    'stochastic_whdbpsoa_rukii.csv',
    'stochastic_whdbpsoa_fatma.csv', 
    'stochastic_whdbpsoa_ryma.csv', 
]

# Initialize an empty list to store dataframes
dfs = []

# Loop through the list of files, read them, and append them to the list
for file in csv_files:
    if os.path.exists(file):  # Check if the file exists
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1')  # Specify encoding
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Concatenate all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('stochastic_whdbpso.csv', index=False)

print("CSV files have been successfully combined into 'stochastic_whdbpso.csv'")
