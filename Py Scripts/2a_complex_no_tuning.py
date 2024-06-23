# 2a - ComplEx

model_name = "ComplEx"

import numpy as np
import os
import csv
import math
import torch
import optuna
import pickle
import datetime
from datetime import datetime
import random
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# LOADING GRAPH from Jupyter Notebook 

import pickle

def calculate_mrr(sorted_indices, true_values):
    true_values_tensor = torch.tensor(true_values)  # Convert NumPy array to PyTorch tensor

    # Find indices of true positive labels
    positive_indices = torch.nonzero(true_values_tensor).squeeze()

    print(f"Are there TRUE LABELS? - {positive_indices.numel()}")

    if positive_indices.numel() == 0:
        return 0.0

    # Map indices in sorted_indices to their ranks --> RANKING
    rank_map = {}
    for rank, idx in enumerate(sorted_indices, start=1):
        rank_map[idx.item()] = rank

    reciprocal_ranks = []
    for idx in positive_indices:
        # print(f"Positive at index - {idx}")
        rank = rank_map.get(idx.item(), 0)
        # print(f"Rank of Positive Index - {rank}")
        # print(f"Adding to ranks - {1.0 / rank}")
        if rank != 0:
            reciprocal_ranks.append(1.0 / rank) # 1/20

    if len(reciprocal_ranks) == 0:
        return 0.0

    # Calculate the mean reciprocal rank
    # print(f"Reciprocal Ranks: {reciprocal_ranks}")
    # print(f"SUM OF RECIPROCAL RANKS - {torch.sum(torch.tensor(reciprocal_ranks, dtype=torch.float))}")
    # print(f"Length of positives in labels - {len(positive_indices)}")
    mrr = torch.sum(torch.tensor(reciprocal_ranks, dtype=torch.float)) / len(positive_indices)
    # print(f"MRR - {mrr}")

    return mrr

print("Started the program...")
# Specify the file path where the data is saved
file_path = "/var/scratch/hwg580/graph_Balanced_HI-Large_Trans.pickle"

# Load the data from the file
with open(file_path, "rb") as f:
    saved_data = pickle.load(f)

print("Loading data...")
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
adjacency_tensor = saved_data['adjacency_tensor']

print("Splitting data...")
# Split the nodes into training, validation, and test sets
num_edges = edges_features.shape[0]
indices = list(range(num_edges))
train_indices, test_val_indices = train_test_split(indices, test_size=0.4, stratify=labels)
val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, stratify=labels[test_val_indices])

print("Creating mask data...")
# Convert indices to a tensor
indices = torch.arange(num_edges)

# Create masks efficiently using torch.zeros_like and indexing
train_mask = torch.zeros_like(indices, dtype=torch.bool)
val_mask = torch.zeros_like(indices, dtype=torch.bool)
test_mask = torch.zeros_like(indices, dtype=torch.bool)

# Set True at indices present in train_indices, val_indices, test_indices
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True
print("Mask data created...")

# GNN
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
        self.adjacency_matrix = adjacency_tensor
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
    
    def forward(self, x, edge_index, edge_attr):
        axw1, ew1 = self.conv1(x, edge_index, edge_attr)
        head_indices, tail_indices = self.mapping(ew1, edge_index)
        scores = self.complex(axw1, ew1, head_indices, tail_indices)
        
        return axw1, ew1, scores # returning x and e embeddings

    def update_edge_attr(self, edge_attr, new_channels):
        num_edge_features = edge_attr.size(1)
        if new_channels > num_edge_features:
            updated_edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.size(0), new_channels - num_edge_features), device=edge_attr.device)), dim=1)
        else:
            updated_edge_attr = edge_attr[:, :new_channels]
        return updated_edge_attr
    
    def complex(self, axw, ew, head_indices, tail_indices):
        heads = axw[head_indices]
        tails = axw[tail_indices]
        # raw_scores = torch.sum(heads * ew * tails, dim=-1)

        # Convert real tensors to complex tensors
        heads = torch.complex(heads, torch.zeros_like(heads))
        ew = torch.complex(ew, torch.zeros_like(ew))
        tails = torch.complex(tails, torch.zeros_like(tails))

        raw_scores = torch.real(torch.sum(heads * ew * torch.conj(tails), dim=-1))
        normalized_scores = torch.sigmoid(raw_scores)  # Apply sigmoid activation
        return normalized_scores

    def mapping(self, ew, edge_index):
        return edge_index[0], edge_index[1]


def assign_predictions(val_scores, threshold=0.5):
    # Assign labels based on a threshold
    predicted_labels = (val_scores >= threshold).float()
    return predicted_labels

# # Now, you can use the best hyperparameters to train your final model
# best_epochs = best_params['epochs']
# best_lr = best_params['lr']
# best_out_channels = best_params['out_channels']
# best_weight_decay = best_params['weight_decay']
# best_dropout = best_params['dropout']
# best_annealing_rate = best_params['annealing_rate']
# annealing_epochs = best_params['annealing_epochs']

# # SAVE hyperparams for ComplEx
# print("Saving Hyperparameters...")
# with open("/var/scratch/hwg580/complex_hyperparams.pickle", "wb") as f:
#     pickle.dump({
#         'best_epochs': best_epochs,
#         'best_lr': best_lr,
#         'best_out_channels': best_out_channels,
#         'best_weight_decay': best_weight_decay,
#         'best_dropout': best_dropout,
#         'best_annealing_rate': best_annealing_rate,
#         'annealing_epochs': annealing_epochs
#     }, f)


# TRAINING
learning_rate = 0.0001
out_channels = 15
weight_decay = 0.00005  # L2 regularization factor
epochs = 50
dropout = 0.1 # dropout probability

# Annealing parameters
annealing_rate = 0.01  # Rate at which to decrease the learning rate
annealing_epochs = 20  # Number of epochs before decreasing learning rate

print("Loading Model...")
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

# Assuming the data loading and model definition parts remain unchanged

# Initialize lists for storing fold-wise metrics
fold_accuracy_list = []
fold_precision_list = []
fold_recall_list = []
fold_f1_list = []
fold_mrr_list = []

# K-fold Cross-Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize variables to track best model
best_model_state = None
best_recall = -1.0  # Initialize to a low value
best_fold_best_metrics = None
best_train_losses = []
best_val_losses = []
best_epoch_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for fold, (train_fold_indices, val_fold_indices) in enumerate(kf.split(range(input_data.edge_attr.shape[0]))):
    print(f"Fold {fold+1}/{k}")
    train_fold_mask = torch.zeros(input_data.edge_attr.shape[0], dtype=torch.bool)
    val_fold_mask = torch.zeros(input_data.edge_attr.shape[0], dtype=torch.bool)

    train_fold_mask[train_fold_indices] = True
    val_fold_mask[val_fold_indices] = True

    # Initialize model, optimizer, and loss function for each fold
    model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    current_fold_best_metrics = {
        'accuracy': -1.0,
        'precision': -1.0,
        'recall': -1.0,
        'f1': -1.0,
        'mrr': -1.0,
        'val_predictions': None,
        'sorted_indices': None
    }

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        x_embedding, e_embedding, scores = model(input_data.x, input_data.edge_index[:, train_fold_mask], input_data.edge_attr[train_fold_mask])
        loss = criterion(scores, labels[train_fold_mask].float())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_x_embedding, val_e_embedding, val_scores = model(input_data.x, input_data.edge_index[:, val_fold_mask], input_data.edge_attr[val_fold_mask])
            val_loss = criterion(val_scores, labels[val_fold_mask].float()).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        val_labels = labels[val_fold_mask]
        val_predictions = assign_predictions(val_scores)

        sorted_indices = torch.argsort(val_scores, descending=True)

        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions)
        val_recall = recall_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions)
        val_mrr = calculate_mrr(torch.argsort(val_scores, descending=True), val_labels)

        print(f"\nEpoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MRR: {val_mrr:.4f}")

        # Store metrics for this epoch
        fold_accuracy_list.append(val_accuracy)
        fold_precision_list.append(val_precision)
        fold_recall_list.append(val_recall)
        fold_f1_list.append(val_f1)
        fold_mrr_list.append(val_mrr)

        # Track best model based on highest recall
        if val_recall > current_fold_best_metrics['recall']:
            current_fold_best_metrics = {
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'mrr': val_mrr,
                'val_predictions': val_predictions,
                'sorted_indices': torch.argsort(val_scores, descending=True)
            }
            # Save the best model state for the current fold
            best_model_state = model.state_dict()

        # Append metrics for best performing epoch (fold)
        if current_fold_best_metrics['recall'] == val_recall:
            best_epoch_metrics['accuracy'].append(val_accuracy)
            best_epoch_metrics['precision'].append(val_precision)
            best_epoch_metrics['recall'].append(val_recall)
            best_epoch_metrics['f1'].append(val_f1)

    # End of epoch loop for the fold

    # Append training and validation losses for best performing epoch (fold)
    best_train_losses.append(train_losses)
    best_val_losses.append(val_losses)

    # Update overall best model based on highest recall across all folds
    if current_fold_best_metrics['recall'] > best_recall:
        best_recall = current_fold_best_metrics['recall']
        best_fold_best_metrics = current_fold_best_metrics

# End of fold loop

# Print best metrics across all folds
if best_fold_best_metrics is not None:
    print("\nBest Model Metrics:")
    print(f"Best Accuracy: {best_fold_best_metrics['accuracy']:.4f}")
    print(f"Best Precision: {best_fold_best_metrics['precision']:.4f}")
    print(f"Best Recall: {best_fold_best_metrics['recall']:.4f}")
    print(f"Best F1 Score: {best_fold_best_metrics['f1']:.4f}")
    print(f"Best MRR: {best_fold_best_metrics['mrr']:.4f}")

# Calculate average metrics across all folds
avg_accuracy = np.mean(fold_accuracy_list)
avg_precision = np.mean(fold_precision_list)
avg_recall = np.mean(fold_recall_list)
avg_f1 = np.mean(fold_f1_list)
avg_mrr = np.mean(fold_mrr_list)

print(f"\nAverage Metrics Across {k} Folds:")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average MRR: {avg_mrr:.4f}")

# Save the best model based on the highest recall score
if best_model_state is not None:
    torch.save(best_model_state, f'/var/scratch/hwg580/{model_name}_best.pt')
    print(f"\nBest model saved with highest recall: {best_recall:.4f}")

# Plotting graphs for best performing model (fold)

# Plot Training and Validation Losses
epoch_numbers = list(range(1, len(best_train_losses[0]) + 1))  # Assuming all folds have the same number of epochs

import os

# Define the results folder path
results_folder = f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/Validation'
os.makedirs(results_folder, exist_ok=True)

# Plot Training and Validation Losses Over Epochs
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(epoch_numbers, best_train_losses[i], label=f"Fold {i+1} Training Loss")
    plt.plot(epoch_numbers, best_val_losses[i], label=f"Fold {i+1} Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training and Validation Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_folder, 'training_validation_losses.png'))
plt.show()

# Plot Accuracy, Precision, Recall, and F1 Score Over Epochs
plt.figure(figsize=(14, 10))

# Accuracy
plt.subplot(2, 2, 1)
for i in range(k):
    plt.plot(best_epoch_metrics['accuracy'][i], label=f"Fold {i+1} Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# Precision
plt.subplot(2, 2, 2)
for i in range(k):
    plt.plot(best_epoch_metrics['precision'][i], label=f"Fold {i+1} Precision")
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision Over Epochs')
plt.legend()

# Recall
plt.subplot(2, 2, 3)
for i in range(k):
    plt.plot(best_epoch_metrics['recall'][i], label=f"Fold {i+1} Recall")
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Over Epochs')
plt.legend()

# F1 Score
plt.subplot(2, 2, 4)
for i in range(k):
    plt.plot(best_epoch_metrics['f1'][i], label=f"Fold {i+1} F1 Score")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'metrics_over_epochs.png'))
plt.show()


import now 
def evaluate_model(predictions, true_values, sorted_indices, mask, model_name):
    # Define the results folder path
    results_folder = f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/Testing'
    os.makedirs(results_folder, exist_ok=True)
    
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

    # Get current datetime for filenames
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    roc_curve_path = os.path.join(results_folder, f'roc_curve_{current_datetime}.png')
    plt.savefig(roc_curve_path)
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(results_folder, f'confusion_matrix_{current_datetime}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    return metrics_dict

metrics_dict = evaluate_model(val_predictions, labels, sorted_indices, val_mask, model_name)

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

best_model_state = torch.load(f'/var/scratch/hwg580/{model_name}_best.pt')
model.load_state_dict(best_model_state)

print("Testing...")
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
metrics_dict = evaluate_model(test_predictions, labels, sorted_indices, test_mask, model_name)

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

# LOGGING
# LOGGING
# Function to log the experiment
def log_experiment(model_name, learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr):
    # Create a folder for the experiment if it doesn't exist
    folder_name = f"/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save metrics and other information to a file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{folder_name}/run_{timestamp}.txt"
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
    print("Updating csv file...")
    csv_file = f"/home/hwg580/thesis/AML-fraud-detector/general.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Timestamp", "learning_rate", "out_channels", "Epoch", "Weight_decay", "Dropout", "Loss", "Accuracy", "Precision", "Recall", "F1 Score", "MRR"])
        writer.writerow([model_name, timestamp,  learning_rate, out_channels, epoch, weight_decay, dropout, loss, accuracy, precision, recall, f1, mrr])

# Inside the training loop, after each epoch:
# Log the experiment
print("Logging...")
log_experiment(model_name=model_name, learning_rate=learning_rate, out_channels=out_channels, epoch=epochs, weight_decay=weight_decay, dropout=dropout, loss=test_loss, accuracy=test_accuracy, precision=test_precision, recall=test_recall, f1=test_f1, mrr=test_mrr)

# PYTORCH.save --> save the tensor for predictions for the graph
torch.save({'test_labels': test_labels}, f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/labels.pt')
torch.save({'predictions': test_predictions}, f'/home/hwg580/thesis/AML-fraud-detector/Results/{model_name}/predictions_{model_name}_{random.randint(1, 100)}.pt')

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
