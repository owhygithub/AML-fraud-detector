# 1c - DISMULT - T+W

model_name = "DisMult-T+W"

import numpy as np
import pickle
import random
import os
import math
import csv
import datetime
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# LOADING GRAPH from Jupyter Notebook 

import pickle

print("Started the program...")
# Specify the file path where the data is saved
file_path = "/var/scratch/hwg580/graph.pickle"

# Load the data from the file
with open(file_path, "rb") as f:
    saved_data = pickle.load(f)

# Now, you can access the saved data using the keys used during saving
dataset = saved_data['dataset']
edges_features = saved_data['edges_features']
links = saved_data['links']
labels = saved_data['labels']
graph_full = saved_data['graph_full']
adjacency_matrix = saved_data['adjacency_matrix']
visual = saved_data['visual']
node_features = saved_data['node_features']
x = saved_data['x']
y = saved_data['y']
input_data = saved_data['input_data']
time_closeness = saved_data['time_closeness']
adjacency_tensor = saved_data['adjacency_tensor']

# Split the nodes into training, validation, and test sets
num_edges = edges_features.shape[0]
indices = list(range(num_edges))
print(indices)
train_indices, test_val_indices = train_test_split(indices, test_size=0.4, stratify=labels)
val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, stratify=labels[test_val_indices])
# Create masks
train_mask = torch.tensor([i in train_indices for i in range(num_edges)], dtype=torch.bool)
val_mask = torch.tensor([i in val_indices for i in range(num_edges)], dtype=torch.bool)
test_mask = torch.tensor([i in test_indices for i in range(num_edges)], dtype=torch.bool)
val_mask

# GRAPH NEURAL NETWORKS
class GNNLayer(MessagePassing):
    def __init__(self, node_features, edge_features, out_channels, dropout):
        super(GNNLayer, self).__init__(aggr='add')
        self.node_features = node_features
        self.edge_features = edge_features
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        
        # Learnable parameters
        self.weight_node = nn.Parameter(torch.Tensor(node_features, out_channels))
        self.weight_edge = nn.Parameter(torch.Tensor(edge_features, out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_node)
        nn.init.xavier_uniform_(self.weight_edge)
        
    def forward(self, x, edge_index, edge_attr):
        # AXW0 + EW1
        global adjacency_tensor
        self.adjacency_matrix = adjacency_matrix
        
        axw = torch.sparse.mm(self.adjacency_matrix, x) @ self.weight_node
        ew = torch.matmul(edge_attr, self.weight_edge)

        axw = self.dropout(axw)  # Apply dropout to node features
        ew = self.dropout(ew)    # Apply dropout to edge features

        return axw, ew

    def update(self, aggr_out):
        return aggr_out
    
class GNNModel(nn.Module):
    def __init__(self, node_features, edge_features, out_channels, dropout):
        super(GNNModel, self).__init__()
        self.conv1 = GNNLayer(node_features, edge_features, out_channels, dropout)

        # Learnable parameters
        self.learnable_weight = nn.Parameter(torch.Tensor(1))
        print(self.learnable_weight.size())
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.learnable_weight, a=0, b=1) 
    
    def forward(self, x, edge_index, edge_attr):
        axw1, ew1 = self.conv1(x, edge_index, edge_attr)

        head_indices, tail_indices = self.mapping(ew1, edge_index)
        # scores = self.dismult(axw1, ew1, head_indices, tail_indices)
        scores = self.dismult(axw1, ew1, head_indices, tail_indices, time_closeness) # add the timestamp
        
        return axw1, ew1, scores # returning x and e embeddings

    def update_edge_attr(self, edge_attr, new_channels):
        num_edge_features = edge_attr.size(1)
        if new_channels > num_edge_features:
            updated_edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.size(0), new_channels - num_edge_features), device=edge_attr.device)), dim=1)
        else:
            updated_edge_attr = edge_attr[:, :new_channels]
        return updated_edge_attr
    
    def dismult(self, axw, ew, head_indices, tail_indices, time_closeness):
        scores = []
        heads = []
        tails = []
        relations = []
        for i in range(ew.size()[0]): # going through all triples
            t = time_closeness[i]
            head = axw[head_indices[i]]
            tail = axw[tail_indices[i]]
            relation = ew[i]
            heads.append(head)
            tails.append(tail)
            relations.append(relation)
            raw_score = torch.sum(head * relation * tail, dim=-1) * (t * self.learnable_weight[0])
            # print(raw_score)
            normalized_score = torch.sigmoid(raw_score)  # Apply sigmoid activation
            scores.append(raw_score) # calc scores
        scores = torch.stack(scores)
        return scores

    def complex(self, axw, ew, head_indices, tail_indices, time_closeness):
        scores = []
        heads = []
        tails = []
        relations = []
        for i in range(ew.size()[0]): # going through all triples
            t = time_closeness[i]
            head = axw[head_indices[i]]
            tail = axw[tail_indices[i]]
            
            relation = ew[i]
            heads.append(head)
            tails.append(tail)
            relations.append(relation)
            # ComplEx
            # print(self.learnable_weight.size())
            # print(t)
            print(self.learnable_weight[0])
            raw_score = torch.real(torch.sum(head * relation * torch.conj(tail), dim=0)) * (t * self.learnable_weight[0]) # TODO add a learnable element to the function for better performance 
            
            normalized_score = torch.sigmoid(raw_score)  # Apply sigmoid activation
            scores.append(raw_score) # calc scores
        scores = torch.stack(scores)
        return scores
    
    def mapping(self, ew, edge_index):
        head_indices = []
        tail_indices = []
        for c in range(ew.size()[0]): # getting all indices
            head_index = edge_index[0][c]
            tail_index = edge_index[1][c]
            head_indices.append(head_index)
            tail_indices.append(tail_index)
        
        return head_indices, tail_indices
    
# Get Hyperparams
file_path = "/var/scratch/hwg580/distmult_hyperparams.pickle"

# Load the data from the file
with open(file_path, "rb") as f:
    saved_data = pickle.load(f)

# Now, you can access the saved data using the keys used during saving
best_epochs = saved_data['best_epochs']
best_lr = saved_data['best_lr']
best_out_channels = saved_data['best_out_channels']
best_weight_decay = saved_data['best_weight_decay']
best_dropout = saved_data['best_dropout']
best_annealing_rate = saved_data['best_annealing_rate']
annealing_epochs = saved_data['annealing_epochs']

# Hyperparams --- adjust to model best hyperparams
learning_rate = best_lr
out_channels = best_out_channels
weight_decay = best_weight_decay  # L2 regularization factor
epochs = best_epochs
dropout = best_dropout # dropout probability

# Annealing parameters
annealing_rate = best_annealing_rate  # Rate at which to decrease the learning rate
annealing_epochs = annealing_epochs  # Number of epochs before decreasing learning rate

model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
def train(data):
    model.train()
    optimizer.zero_grad()
    x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, train_mask], data.edge_attr[train_mask])

    loss = criterion(scores, labels[train_mask].float())
    loss.backward()
    optimizer.step()
    
    return loss.item(), x_embedding, e_embedding, scores

# Validation function
def validate(data):
    model.eval()
    with torch.no_grad():
        x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, val_mask], data.edge_attr[val_mask])
        val_loss = criterion(scores, labels[val_mask].float()).item()
    return x_embedding, e_embedding, scores, val_loss

# Test function
def test(data):
    model.eval()
    with torch.no_grad():
        x_embedding, e_embedding, scores = model(data.x, data.edge_index[:, test_mask], data.edge_attr[test_mask])
        test_loss = criterion(scores, labels[test_mask].float()).item()
    return x_embedding, e_embedding, scores, test_loss
def assign_top_n_predictions(val_scores, val_labels):
    # Sort indices of val_scores in descending order
    sorted_indices = torch.argsort(val_scores, descending=True)

    # Determine the number of top predictions to assign as 1
    num_ones = int(torch.sum(val_labels).item())

    # Create a tensor to store the predicted labels (initialized with all zeros)
    predicted_labels = torch.zeros_like(val_labels)

    # Assign the top n predictions as 1
    predicted_labels[sorted_indices[:num_ones]] = 1

    return predicted_labels, sorted_indices
def assign_predictions(val_scores, threshold=0.5):
    # Assign labels based on a threshold
    predicted_labels = (val_scores >= threshold).float()
    return predicted_labels
def calculate_mrr(sorted_indices, true_values):
    # print("Calculating MRR...")
    rank = 0
    count = 0
    for i in range( len(true_values) ):
        if true_values[i] == 1: # should only be the positive ones --> 600
            count += 1
            for j in range( len(sorted_indices) ):
                if sorted_indices[j] == i:
                    # print(f"sorted indices: {sorted_indices[j]}")
                    # print(f"i: {i}")
                    position = j+1
                    # print(f"position: {position}")
                    break
            rank += 1/position
            # print(f"rank - individual: {rank}")
    # print(f"rank: {rank}")
    # print(f"n of true values: {count}")
    mrr = rank / count
    # print("MRR Calculated...")
    return mrr
# Continue training loop from provided script
losses = []
val_losses = []
best_val_loss = float('inf')
patience = 5

all_x_embeddings = []
all_e_embeddings = []

# Storage for metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for epoch in range(epochs):
    # Adjust learning rate based on annealing schedule
    if epoch % annealing_epochs == 0 and epoch != 0:
        new_learning_rate = learning_rate * math.exp(-annealing_rate * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
            
    loss, x_embedding, e_embedding, scores = train(input_data)
    val_x_embedding, val_e_embedding, val_scores, val_loss = validate(input_data)

    losses.append(loss)
    val_losses.append(val_loss)

    all_x_embeddings.append(x_embedding.detach().cpu().numpy())
    all_e_embeddings.append(e_embedding.detach().cpu().numpy())
    # print(f"This is Fraudulent - {scores[8000]}")
    # print(f"This is Not fraudulent - {scores[2000]}")

    val_labels = labels[val_mask]
    # predictions, sorted_indices = assign_top_n_predictions(val_scores, val_labels)
    predictions = assign_predictions(val_scores)
    sorted_indices = torch.argsort(val_scores, descending=True)

    # Calculate evaluation metrics
    accuracy = accuracy_score(val_labels, predictions)
    precision = precision_score(val_labels, predictions)
    recall = recall_score(val_labels, predictions)
    f1 = f1_score(val_labels, predictions)

    # Append metrics to respective lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"\nEpoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    if epoch % 10 == 0 and epoch != 0:
        # calculate MRR
        # mrr = calculate_mrr(sorted_indices, val_labels)
        mrr = calculate_mrr(sorted_indices, val_labels) # TODO MRR USING SORTED INDICES? WHY?
        print(f"This is the MRR for epoch {epoch}: {mrr}")
    # EARLY STOPPING CHECK 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the model if validation loss improves
        torch.save(model.state_dict(), f'/var/scratch/hwg580/{model_name}.pt')
    else:
        patience_counter += 1
        if patience_counter > patience:
            print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
            break
val_scores.size()
# Plot
epoch_numbers = list(range(1, len(losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epoch_numbers, losses, label="Training Loss")
plt.plot(epoch_numbers, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training and Validation Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 8))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(accuracy_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Precision
plt.subplot(2, 2, 2)
plt.plot(precision_list, label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(2, 2, 3)
plt.plot(recall_list, label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# F1 Score
plt.subplot(2, 2, 4)
plt.plot(f1_list, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
def evaluate_model(predictions, true_values, sorted_indices, mask):

    true_values = true_values[mask].float()
    # Convert tensors to numpy arrays
    predictions = predictions.cpu().numpy()
    true_values = true_values.cpu().numpy()
    
    mrr = calculate_mrr(sorted_indices, true_values) # TODO --> this is just the accuracy now ... 
    print(f"MRR = {mrr}")
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions)
    recall = recall_score(true_values, predictions)
    f1 = f1_score(true_values, predictions)
    cm = confusion_matrix(true_values, predictions)
    classification_rep = classification_report(true_values, predictions)
    
    # ROC Curve and AUC if applicable
    try:
        fpr, tpr, thresholds = roc_curve(true_values, predictions)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        fpr, tpr, roc_auc = None, None, None
    
    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "Classification Report": classification_rep,
        "ROC Curve": (fpr, tpr, roc_auc)
    }

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return metrics_dict
metrics_dict = evaluate_model(predictions, labels, sorted_indices, val_mask)

# Print Evaluation Metrics
print("Evaluation Metrics:")
print("-------------------\n")
for metric_name, metric_value in metrics_dict.items():
    if metric_name == "Confusion Matrix":
        print("Confusion Matrix:")
        print(metric_value)
    elif metric_name == "Classification Report":
        print("Classification Report:")
        print(metric_value)
    elif metric_name == "ROC Curve":
        fpr, tpr, roc_auc = metric_value
        print("ROC Curve:")
        print("- False Positive Rate:", fpr)
        print("- True Positive Rate:", tpr)
        print("- AUC:", roc_auc)
    else:
        print(f"{metric_name}: {metric_value}")
    print()
# TEST DATA
test_x_embedding, test_e_embedding, test_scores, test_loss = test(input_data)
print(f"Test Loss: {test_loss:.4f}")

test_labels = labels[test_mask]
test_predictions = assign_predictions(test_scores)

sorted_indices = torch.argsort(test_scores, descending=True)

# Calculate evaluation metrics
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
test_recall = recall_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)

test_mrr = calculate_mrr(sorted_indices, test_labels)

print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")
print(f"This is the MRR testing data: {test_mrr}")
metrics_dict = evaluate_model(test_predictions, labels, sorted_indices, test_mask)

# Print Evaluation Metrics
print("Evaluation Metrics:")
print("-------------------\n")
for metric_name, metric_value in metrics_dict.items():
    if metric_name == "Confusion Matrix":
        print("Confusion Matrix:")
        print(metric_value)
    elif metric_name == "Classification Report":
        print("Classification Report:")
        print(metric_value)
    elif metric_name == "ROC Curve":
        fpr, tpr, roc_auc = metric_value
        print("ROC Curve:")
        print("- False Positive Rate:", fpr)
        print("- True Positive Rate:", tpr)
        print("- AUC:", roc_auc)
    else:
        print(f"{metric_name}: {metric_value}")
    print()

# Function to log the experiment
def log_experiment(model_name, learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr):
    # Save metrics and other information to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"/var/scratch/hwg580/run_{model_name}_{timestamp}.txt"
    with open(file_name, "w") as f:
        f.write(f"-- HYPERPARAMS:\n")
        f.write(f"TimeStamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"out_channels: {out_channels}\n")
        f.write(f"Epoch: {epoch}\n\n")
        f.write(f"Weight_decay: {weight_decay}\n\n")
        f.write(f"Dropout: {dropout}\n\n")
        f.write(f"-- RESULTS:\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"MRR: {mrr}\n")
    
    # Update the general CSV file
    csv_file = f"/var/scratch/hwg580/general.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Timestamp", "learning_rate", "out_channels", "Epoch", "Weight_decay", "Dropout", "Loss", "Accuracy", "Precision", "Recall", "F1 Score", "MRR"])
        writer.writerow([model_name, timestamp,  learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr])

# Inside the training loop, after each epoch:
# Log the experiment
log_experiment(model_name=model_name, learning_rate=learning_rate, out_channels=out_channels, epoch=epoch, weight_decay=weight_decay, dropout=dropout, loss=test_loss, accuracy=test_accuracy, precision=test_precision, recall=test_recall, f1=test_f1, mrr=test_mrr)

# PYTORCH.save --> save the tensor for predictions for the graph
torch.save({'test_labels': test_labels}, f'Results/{model_name}/labels.pt')
torch.save({'predictions': test_predictions}, f'Results/{model_name}/predictions_{model_name}_{random.randint(1, 100)}.pt')
# Save in RDF format
# import gzip
# import torch

# def save_embeddings_as_triples(edge_index, node_embeddings, edge_embeddings, file_path):
#     with gzip.open(file_path, 'wt') as f:
#         for i in range(edge_index.size(1)):
#             head_index = edge_index[0, i].item()
#             tail_index = edge_index[1, i].item()
#             head_uri = f"http://example.org/node/{head_index}"
#             tail_uri = f"http://example.org/node/{tail_index}"
#             relation_uri = f"http://example.org/relation/{i}"

#             # Write connectedTo triple
#             f.write(f'<{head_uri}> <http://example.org/ontology#connectedTo> <{tail_uri}>.\n')
            
#             # Write head node embedding triple
#             head_embedding = node_embeddings[head_index].numpy()
#             head_str = ' '.join(map(str, head_embedding))
#             f.write(f'<{head_uri}> <http://example.org/ontology#hasEmbedding> "{head_str}."\n')
            
#             # Write tail node embedding triple
#             tail_embedding = node_embeddings[tail_index].numpy()
#             tail_str = ' '.join(map(str, tail_embedding))
#             f.write(f'<{tail_uri}> <http://example.org/ontology#hasEmbedding> "{tail_str}".\n')
            
#             # Write edge embedding triple
#             relation_embedding = edge_embeddings[i].numpy()
#             relation_str = ' '.join(map(str, relation_embedding))
#             f.write(f'<{relation_uri}> <http://example.org/ontology#hasEmbedding> "{relation_str}".\n')

# edge_index_train_triples = input_data.edge_index[:, train_mask]

# # Example usage after training
# x_embeddings = torch.tensor(all_x_embeddings[-1])  # Use the last epoch's embeddings
# e_embeddings = torch.tensor(all_e_embeddings[-1])  # Use the last epoch's embeddings

# save_embeddings_as_triples(edge_index_train_triples, x_embeddings, e_embeddings, f"Saved-Data/rdf_triples_{model_name}.nt.gz")
