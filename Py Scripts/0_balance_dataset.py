import dask.dataframe as dd
import numpy as np

print("Import Successful...")

print("Loading...")
# Load the data with Dask
filename = '/var/scratch/hwg580/HI-Large_Trans.csv'
data = dd.read_csv(filename, parse_dates=['Timestamp'], infer_datetime_format=True)

# Identify fraudulent transactions and connected non-fraudulent transactions efficiently
print("Creating balanced dataset...")

# Find fraudulent transactions indices
fraud_indices = data[data['Is Laundering'] == 1].index

# Find connected non-fraudulent transactions using a set for faster lookup
print("Identify non-fraudulent transactions connected to fraudulent ones ...")
connected_accounts = set(data.loc[fraud_indices, 'Account'].unique().compute()).union(set(data.loc[fraud_indices, 'To Bank'].unique().compute()))

# Filter non-fraudulent transactions that are connected to fraudulent ones
print("Filter non-fraudulent transactions that are connected to fraudulent ones ...")
connected_non_fraud_indices = data[(data['Is Laundering'] == 0) & (data['Account'].isin(connected_accounts) | data['To Bank'].isin(connected_accounts))].index

# Combine indices of fraudulent and connected non-fraudulent transactions
print("Combine fraudulent and connected non-fraudulent transactions ...")
combined_indices = dd.concat([fraud_indices, connected_non_fraud_indices]).compute()

# Sample from combined indices to create a balanced dataset
print("Ensure the final dataset size is between 3 to 5 million transactions ...")
np.random.seed(42)  # Set random seed for reproducibility
final_size = min(max(len(combined_indices), 3000000), 5000000)
sampled_indices = np.random.choice(combined_indices, size=final_size, replace=False)

# Create balanced dataset
balanced_data = data.loc[sampled_indices].compute().reset_index(drop=True)

# Ensure 10% of the dataset is fraudulent
print("Ensure 10 percent of the dataset is fraudulent ...")
fraud_count = int(final_size * 0.1)
fraudulent_sample = balanced_data[balanced_data['Is Laundering'] == 1].sample(n=fraud_count, random_state=42)
non_fraudulent_sample = balanced_data[balanced_data['Is Laundering'] == 0].sample(n=final_size - fraud_count, random_state=42)

balanced_data = dd.concat([fraudulent_sample, non_fraudulent_sample]).sample(frac=1, random_state=42).compute().reset_index(drop=True)

print("Saving data ...")
# Save the balanced dataset
balanced_data.to_csv("/var/scratch/hwg580/Balanced_HI-Large_Trans.csv", index=False, single_file=True)

print("Balanced dataset created and saved to 'Balanced_HI-Large_Trans.csv'.")

print("Computing Statistics...")
# Compute statistics using Dask dataframe operations
total_transactions = data.index.size.compute()
fraudulent_transactions = data['Is Laundering'].sum().compute()
fraud_percentage = (fraudulent_transactions / total_transactions) * 100

# Compute percentages of different currencies
currency_counts = data['Receiving Currency'].value_counts().compute()
currency_percentage = (currency_counts / currency_counts.sum()) * 100

# Compute percentages of different payment formats
payment_format_counts = data['Payment Format'].value_counts().compute()
payment_format_percentage = (payment_format_counts / payment_format_counts.sum()) * 100

# Print statistics
print(f"Total transactions: {total_transactions}")
print(f"Fraudulent transactions: {fraudulent_transactions}")
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
print("\nPercentage of different currencies:")
print(currency_percentage)
print("\nPercentage of different payment formats:")
print(payment_format_percentage)
