import pandas as pd
import numpy as np

# Load the data
filename = f'/var/scratch/hwg580/HI-Large_Trans.csv'
data = pd.read_csv(filename)

# Convert the 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Compute statistics
total_transactions = len(data)
fraudulent_transactions = data['Is Laundering'].sum()
fraud_percentage = (fraudulent_transactions / total_transactions) * 100

currency_counts = data['Receiving Currency'].value_counts(normalize=True) * 100
payment_format_counts = data['Payment Format'].value_counts(normalize=True) * 100

# Print statistics
print(f"Total transactions: {total_transactions}")
print(f"Fraudulent transactions: {fraudulent_transactions}")
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
print("\nPercentage of different currencies:")
print(currency_counts)
print("\nPercentage of different payment formats:")
print(payment_format_counts)

# Create a balanced dataset
# Step 1: Identify all fraudulent transactions
fraud_data = data[data['Is Laundering'] == 1]

# Step 2: Identify non-fraudulent transactions connected to fraudulent ones
# Create sets to hold the connected accounts
connected_accounts = set(fraud_data['Account'].unique()).union(set(fraud_data['To Bank'].unique()))

# Filter non-fraudulent transactions that are connected to fraudulent ones
connected_non_fraud_data = data[(data['Is Laundering'] == 0) & (data['Account'].isin(connected_accounts) | data['To Bank'].isin(connected_accounts))]

# Combine fraudulent and connected non-fraudulent transactions
combined_data = pd.concat([fraud_data, connected_non_fraud_data])

# Ensure the final dataset size is between 3 to 5 million transactions
final_size = min(max(len(combined_data), 3000000), 5000000)
sampled_data = combined_data.sample(n=final_size, random_state=42)

# Ensure 10% of the dataset is fraudulent
final_fraud_count = int(0.1 * final_size)
final_non_fraud_count = final_size - final_fraud_count

fraud_transactions = sampled_data[sampled_data['Is Laundering'] == 1]
non_fraud_transactions = sampled_data[sampled_data['Is Laundering'] == 0]

if len(fraud_transactions) > final_fraud_count:
    fraud_transactions = fraud_transactions.sample(n=final_fraud_count, random_state=42)
if len(non_fraud_transactions) > final_non_fraud_count:
    non_fraud_transactions = non_fraud_transactions.sample(n=final_non_fraud_count, random_state=42)

balanced_data = pd.concat([fraud_transactions, non_fraud_transactions]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv("/var/scratch/hwg580/Balanced_HI-Large_Trans_balanced.csv", index=False)

print("Balanced dataset created and saved to 'Balanced_HI-Large_Trans.csv'.")
