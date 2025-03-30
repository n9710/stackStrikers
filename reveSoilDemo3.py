import pandas as pd
import re

# Load the CSV file
def process_soil_data(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure "Records" column is treated as a string
    df["Records"] = df["Records"].astype(str)
    
    # Extract base sample name (e.g., "100_0ml" from "100_0ml-1")
    df["Sample_Group"] = df["Records"].apply(lambda x: re.sub(r"-\d+$", "", x))
    
    # Convert numeric columns explicitly
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Group by the extracted base sample name and calculate the mean
    df_grouped = df.groupby("Sample_Group")[numeric_cols].mean()
    
    # Reset index for a cleaner format
    df_grouped.reset_index(inplace=True)
    
    # Display the transformed DataFrame in the console
    print("Transformed DataFrame:")
    print(df_grouped.head())  # Display first few rows
    
    # Export to CSV
    # df_grouped.to_csv(output_path, index=False)
    # print(f"Transformed data saved to {output_path}")
    
    df_grouped.to_csv(output_path)

# File path (update accordingly)
file_path = "Sheet1.csv"  # Change this if needed
output_path = "transformed_soildataset.csv"

df_transformed = process_soil_data(file_path, output_path)
