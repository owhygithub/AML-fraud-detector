import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from functions import *

class AMLDataPreprocessing:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.labels = None
        self.node_features = None
        self.edges_features = None
        self.links = None
        self.graph_full = None
        self.adjacency_matrix = None
        self.edge_index = None
        self.input_data = None

    def process_data(self):
        # BASICS
        self.data = pd.read_csv(self.filename)
        laundering_accounts = list(self.data[self.data["Is Laundering"]==1]["Account"])
        self.labels = self.data["Is Laundering"].to_numpy()
        merged_unique_accounts = pd.concat([self.data["Account"], self.data["Account.1"]]).unique()

        print(f"Shape of DataFrame - {self.data.shape}")
        print("---- info ----")
        print(self.data.info())
        print("---- basic calculations ----")
        print(self.data.describe())
        print("Null elements?")
        print(self.data.isnull().sum())
        print("Fraudulent or Not? - Y labels")
        print(f"Number of fraudulent transactions - {len(self.data[self.data['Is Laundering']==1])}")
        print(f"Number of non-fraudulent transactions - {len(self.data[self.data['Is Laundering']==0])}")
        print(f"Laundering Accounts - {laundering_accounts}")
        print("Are Amount Paid entirely equal to Amount Received?\n - " + str(self.data["Amount Paid"].equals(self.data["Amount Received"])))
        print("Are Currency Received entirely equal to Currency Paid?\n - " + str(self.data["Payment Currency"].equals(self.data["Receiving Currency"])))
        print(sorted(self.data["Receiving Currency"].unique()))
        print(sorted(self.data["Payment Currency"].unique()))
        print(sorted(self.data["Payment Format"].unique()))
        print(f"Number of unique accounts: {len(merged_unique_accounts)}")

        # NODE MATRIX
        unique_accounts = get_nodes(self.data)
        node_features = one_hot_encoding(unique_accounts, column="Currency")
        from_bank_col = node_features.pop('Bank')
        account_col = node_features.pop('Accounts')
        node_labels = pd.DataFrame(account_col)

        

        # EDGE MATRIX



    def create_full_graph(self):
        graph_full = nx.Graph()
        for link in self.links:
            graph_full.add_edge(link['source'], link['destination'])
        return graph_full

    def visualize_graph(self, graph):
        pos = nx.random_layout(graph)
        plt.figure(figsize=(25, 15))
        nx.draw(
            graph,
            pos,
            node_size=300,
            with_labels=True,
            font_size=7,
            font_weight='bold',
            node_color='lightblue',
            edge_color='gray',
            width=1,
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
        )
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            label_pos=0.5,
            font_size=7,
            font_color='green',
        )
        plt.title(f'Graph Visualization of all transactions')
        plt.axis('off')
        plt.show()