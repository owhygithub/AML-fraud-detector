import pickle
import torch
import numpy as np
import networkx as nx
from functions1 import AMLDataPreprocessing
from scipy.sparse import coo_matrix

print("Import Successful...")

dataset = "HI-Small_Trans"
filename = f'/var/scratch/hwg580/{dataset}.csv'

# Create an instance of the AMLDataPreprocessing class
print("Initializing AML Data Preprocessing...")
data_preprocessor = AMLDataPreprocessing(filename)
# Process the data
print("Processing data...")
input_data, graph_full, x, y, labels, links, edges_amount, node_features, edges_features, time_closeness = data_preprocessor.process_data()
print("Data Processed Successfully!")

# Visualize
visual = data_preprocessor.visualize_graph(links, labels)

# Convert to a boolean adjacency matrix
adjacency_matrix = nx.adjacency_matrix(graph_full).astype(bool)
print(adjacency_matrix)
print(f"Size of adjacency_matrix: {adjacency_matrix.size()}")

print(f"input data: {input_data}")

with open("/var/scratch/hwg580/Saved-Data/graph.pickle", "wb") as f:
    pickle.dump({
        'dataset': dataset,
        'visual': visual,
        'edges_features': edges_features,
        'links': links,
        'graph_full': graph_full,
        'adjacency_matrix': adjacency_matrix,
        'node_features': node_features,
        'labels': labels,
        'input_data': input_data,
        'x': x,
        'y': y,
        'time_closeness': time_closeness
    }, f)
