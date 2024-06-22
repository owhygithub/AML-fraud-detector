import pandas as pd
import numpy as np

print("Import Successful...")

print("Loading...")
# Load the data with memory-efficient settings
filename = '/var/scratch/hwg580/HI-Large_Trans.csv'
data = pd.read_csv(filename, parse_dates=['Timestamp'], infer_datetime_format=True)

print("Computing Statistics...")
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

# Identify fraudulent transactions and connected non-fraudulent transactions efficiently
print("Creating balanced dataset...")
fraud_indices = data.index[data['Is Laundering'] == 1]

# Find connected non-fraudulent transactions using a set for faster lookup
print("Identify non-fraudulent transactions connected to fraudulent ones ...")
connected_accounts = set(data.loc[fraud_indices, 'Account'].unique()).union(set(data.loc[fraud_indices, 'To Bank'].unique()))

# Filter non-fraudulent transactions that are connected to fraudulent ones
print("Filter non-fraudulent transactions that are connected to fraudulent ones ...")
connected_non_fraud_indices = data.index[(data['Is Laundering'] == 0) & (data['Account'].isin(connected_accounts) | data['To Bank'].isin(connected_accounts))]

# Combine indices of fraudulent and connected non-fraudulent transactions
print("Combine fraudulent and connected non-fraudulent transactions ...")
combined_indices = np.concatenate((fraud_indices, connected_non_fraud_indices))

# Sample from combined indices to create a balanced dataset
print("Ensure the final dataset size is between 3 to 5 million transactions ...")
np.random.seed(42)  # Set random seed for reproducibility
final_size = min(max(len(combined_indices), 3000000), 5000000)
sampled_indices = np.random.choice(combined_indices, size=final_size, replace=False)

# Create balanced dataset
balanced_data = data.loc[sampled_indices].reset_index(drop=True)

# Ensure 10% of the dataset is fraudulent
print("Ensure 10 percent of the dataset is fraudulent ...")
fraud_count = int(final_size * 0.1)
fraudulent_sample = balanced_data[balanced_data['Is Laundering'] == 1].sample(n=fraud_count, random_state=42)
non_fraudulent_sample = balanced_data[balanced_data['Is Laundering'] == 0].sample(n=final_size - fraud_count, random_state=42)

balanced_data = pd.concat([fraudulent_sample, non_fraudulent_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Saving data ...")
# Save the balanced dataset
balanced_data.to_csv("/var/scratch/hwg580/Balanced_HI-Large_Trans.csv", index=False)

print("Balanced dataset created and saved to 'Balanced_HI-Large_Trans.csv'.")