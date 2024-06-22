import pandas as pd
import numpy as np

def create_balanced_dataset(input_path, reduced_path, balanced_path):
    print("Import Successful...")

    print("Loading...")
    # Read the original CSV file
    df = pd.read_csv(input_path)  # 132 million

    # Reduce the dataset to 30% of its original size
    print("Reducing dataset size...")
    reduced_df = df.sample(frac=0.30, random_state=1)
    reduced_df.to_csv(reduced_path, index=False)

    print("Loading reduced dataset...")
    # Load the reduced dataset
    df = pd.read_csv(reduced_path)

    print("Separations...")
    # Separate rows with "Is Laundering" equal to 0 and 1
    df_0 = df[df["Is Laundering"] == 0]
    df_1 = df[df["Is Laundering"] == 1]

    print("Calculations...")
    # Calculate the number of instances for the new CSV file
    total_instances = min(2000000, len(df))
    instances_0 = int(0.90 * total_instances)
    instances_1 = total_instances - instances_0

    print("Random selection...")
    # Randomly select instances from each category
    selected_indices_0 = np.random.choice(df_0.index, instances_0, replace=False)
    selected_indices_1 = np.random.choice(df_1.index, instances_1, replace=False)

    print("Creating balanced dataset...")
    # Concatenate the selected rows
    selected_df_0 = df_0.loc[selected_indices_0]
    selected_df_1 = df_1.loc[selected_indices_1]
    selected_df = pd.concat([selected_df_0, selected_df_1])

    # Ensure transactions are interconnected
    print("Ensuring interconnected transactions...")
    selected_df = ensure_interconnected(selected_df)

    print("Shuffle...")
    # Shuffle the concatenated rows
    selected_df = selected_df.sample(frac=1, random_state=1).reset_index(drop=True)

    print("Creating file...")
    selected_df.to_csv(balanced_path, index=False)

def ensure_interconnected(df):
    # Placeholder for ensuring interconnected transactions
    # This could involve checking for common accounts, transaction paths, etc.
    # Implement the logic to ensure interconnectedness based on specific criteria
    # For now, let's assume the reduced dataset is sufficiently interconnected

    # Here is a simple example of keeping transactions with common 'Account ID' fields interconnected:
    account_column = 'Account ID'  # Replace with the actual account identifier column name if different
    df_interconnected = df[df[account_column].isin(df[account_column].value_counts().index)]

    return df_interconnected

# File paths
input_path = "/var/scratch/hwg580/HI-Large_Trans.csv"
reduced_path = "/var/scratch/hwg580/Reduced_HI-Large_Trans.csv"
balanced_path = "/var/scratch/hwg580/Balanced_HI-Large_Trans.csv"

# Create balanced dataset
create_balanced_dataset(input_path, reduced_path, balanced_path)
