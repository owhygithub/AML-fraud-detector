# TODO k-fold cross validation --> move from hyperparam tuning to Training Loop ( for better validation representation --> getting best model --> use for testing )

model_name = "DisMult"

import numpy as np
import os
import csv
import math
import torch
import optuna
import pickle
import random
import datetime
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

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
adjacency_tensor = saved_data['adjacency_tensor']

train_loader = DataLoader([input_data], batch_size=32, shuffle=True)

# Split the nodes into training, validation, and test sets
num_edges = edges_features.shape[0]
indices = list(range(num_edges))
# print(indices)
train_indices, test_val_indices = train_test_split(indices, test_size=0.4, stratify=labels)
val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, stratify=labels[test_val_indices])

# Create masks
train_mask = torch.tensor([i in train_indices for i in range(num_edges)], dtype=torch.bool)
val_mask = torch.tensor([i in val_indices for i in range(num_edges)], dtype=torch.bool)
test_mask = torch.tensor([i in test_indices for i in range(num_edges)], dtype=torch.bool)

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

        # self.threshold = nn.Parameter(torch.tensor([0.5]))  # Trainable threshold parameter
    
    def forward(self, x, edge_index, edge_attr):
        axw1, ew1 = self.conv1(x, edge_index, edge_attr)

        head_indices, tail_indices = self.mapping(ew1, edge_index)
        # scores = self.dismult(axw1, ew1, head_indices, tail_indices)
        scores = self.dismult(axw1, ew1, head_indices, tail_indices)
        
        return axw1, ew1, scores # returning x and e embeddings

    def update_edge_attr(self, edge_attr, new_channels):
        num_edge_features = edge_attr.size(1)
        if new_channels > num_edge_features:
            updated_edge_attr = torch.cat((edge_attr, torch.zeros((edge_attr.size(0), new_channels - num_edge_features), device=edge_attr.device)), dim=1)
        else:
            updated_edge_attr = edge_attr[:, :new_channels]
        return updated_edge_attr
    
    def dismult(self, axw, ew, head_indices, tail_indices):
        scores = []
        heads = []
        tails = []
        relations = []
        for i in range(ew.size()[0]): # going through all triples
            head = axw[head_indices[i]]
            tail = axw[tail_indices[i]]
            relation = ew[i]
            heads.append(head)
            tails.append(tail)
            relations.append(relation)
            raw_score = torch.sum(head * relation * tail, dim=-1)
            # print(raw_score)
            normalized_score = torch.sigmoid(raw_score)  # Apply sigmoid activation
            scores.append(raw_score) # calc scores
        scores = torch.stack(scores)
        return scores

    def complex(self, axw, ew, head_indices, tail_indices):
        scores = []
        heads = []
        tails = []
        relations = []
        for i in range(ew.size()[0]): # going through all triples
            head = axw[head_indices[i]]
            tail = axw[tail_indices[i]]
            relation = ew[i]
            heads.append(head)
            tails.append(tail)
            relations.append(relation)
            # ComplEx
            raw_score = torch.real(torch.sum(head * relation * torch.conj(tail), dim=0)) # TODO add a learnable element to the function for better performance 
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

# HYPERPARAMS
def assign_predictions(val_scores, threshold=0.5):
    # Assign labels based on a threshold
    predicted_labels = (val_scores >= threshold).float()
    return predicted_labels

# Define the objective function for Optuna
# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    epochs = trial.suggest_categorical('epochs', [50, 100])  # Reduced epochs
    lr = trial.suggest_categorical('lr', [0.001, 0.0001])
    out_channels = trial.suggest_categorical('out_channels', [10, 15])
    weight_decay = trial.suggest_categorical('weight_decay', [0.0005, 0.00005])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.5])
    annealing_rate = trial.suggest_categorical('annealing_rate', [0.01, 0.001])
    annealing_epochs = trial.suggest_categorical('annealing_epochs', [10, 20])

    # Initialize model, optimizer, and loss function for each fold
    model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_val_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(input_data.edge_attr.shape[0]))):
        print(f"Fold {fold + 1}/{k}")

        train_mask = torch.tensor([i in train_indices for i in range(input_data.edge_attr.shape[0])], dtype=torch.bool)
        val_mask = torch.tensor([i in val_indices for i in range(input_data.edge_attr.shape[0])], dtype=torch.bool)

        patience_counter = 0
        best_val_loss = float('inf')
        patience = 2

        for epoch in range(epochs):
            if epoch % annealing_epochs == 0 and epoch != 0:
                new_learning_rate = lr * math.exp(-annealing_rate * epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate

            model.train()
            optimizer.zero_grad()
            x_embedding, e_embedding, scores = model(input_data.x, input_data.edge_index[:, train_mask], input_data.edge_attr[train_mask])
            loss = criterion(scores, labels[train_mask].float())
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _, _, val_scores = model(input_data.x, input_data.edge_index[:, val_mask], input_data.edge_attr[val_mask])
                val_loss = criterion(val_scores, labels[val_mask].float()).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    break

            val_labels = labels[val_mask]
            val_predictions = assign_predictions(val_scores)

            val_accuracy = accuracy_score(val_labels.cpu(), val_predictions.cpu())
            val_precision = precision_score(val_labels.cpu(), val_predictions.cpu())
            val_recall = recall_score(val_labels.cpu(), val_predictions.cpu())
            val_f1 = f1_score(val_labels.cpu(), val_predictions.cpu())

            all_val_losses.append(val_loss)
            all_accuracies.append(val_accuracy)
            all_precisions.append(val_precision)
            all_recalls.append(val_recall)
            all_f1s.append(val_f1)

    mean_val_loss = sum(all_val_losses) / len(all_val_losses)
    mean_accuracy = sum(all_accuracies) / len(all_accuracies)
    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)
    mean_f1 = sum(all_f1s) / len(all_f1s)

    return mean_recall

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=32)  # Number of trials can be adjusted

# Print the best hyperparameters
best_params = study.best_params
print("Best hyperparameters found: ", best_params)

# Train and evaluate with the best hyperparameters using k-fold cross-validation
best_epochs = best_params['epochs']
best_lr = best_params['lr']
best_out_channels = best_params['out_channels']
best_weight_decay = best_params['weight_decay']
best_dropout = best_params['dropout']
best_annealing_rate = best_params['annealing_rate']
best_annealing_epochs = best_params['annealing_epochs']

# SAVE hyperparams for DistMult

with open("/var/scratch/hwg580/distmult_hyperparams.pickle", "wb") as f:
    pickle.dump({
        'best_epochs': best_epochs,
        'best_lr': best_lr,
        'best_out_channels': best_out_channels,
        'best_weight_decay': best_weight_decay,
        'best_dropout': best_dropout,
        'best_annealing_rate': best_annealing_rate,
        'annealing_epochs': best_annealing_epochs
    }, f)


# TRAINING
# Hyperparams
learning_rate = best_lr
out_channels = best_out_channels
weight_decay = best_weight_decay  # L2 regularization factor
epochs = best_epochs
dropout = best_dropout # dropout probability

# Annealing parameters
annealing_rate = best_annealing_rate  # Rate at which to decrease the learning rate
annealing_epochs = best_annealing_epochs # Number of epochs before decreasing learning rate

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

# K-fold Cross-Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
patience = 10

# Storage for metrics across folds
fold_accuracy_list = []
fold_precision_list = []
fold_recall_list = []
fold_f1_list = []
fold_mrr_list = []
best_model_state = None
best_val_loss = float('inf')
best_train_losses = []
best_val_losses = []
best_val_accuracies = []
best_epoch_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'mrr': [],
    'train_loss': [],
    'val_loss': []
}

for fold, (train_fold_indices, val_fold_indices) in enumerate(kf.split(range(input_data.edge_attr.shape[0]))):
    print(f"\nFold {fold + 1}/{k}")

    train_fold_mask = torch.tensor([i in train_fold_indices for i in range(input_data.edge_attr.shape[0])], dtype=torch.bool)
    val_fold_mask = torch.tensor([i in val_fold_indices for i in range(input_data.edge_attr.shape[0])], dtype=torch.bool)

    # Initialize model, optimizer, and loss function for each fold
    model = GNNModel(node_features=input_data.x.size(1), edge_features=input_data.edge_attr.size(1), out_channels=out_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_fold_val_loss = float('inf')
    best_fold_train_losses = []
    best_fold_epoch_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mrr': [],
        'train_loss': [],
        'val_loss': [],
        'val_predictions': [],
        'sorted_indices': []
    }
    patience_counter = 0

    for epoch in range(epochs):
        # Adjust learning rate based on annealing schedule
        if epoch % annealing_epochs == 0 and epoch != 0:
            new_learning_rate = learning_rate * math.exp(-annealing_rate * epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate

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

        train_labels = labels[train_fold_mask]
        val_labels = labels[val_fold_mask]

        # Calculate metrics
        train_predictions = assign_predictions(scores)
        val_predictions = assign_predictions(val_scores)

        sorted_indices = torch.argsort(val_scores, descending=True)

        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions)
        val_recall = recall_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions)
        val_mrr = calculate_mrr(torch.argsort(val_scores, descending=True), val_labels)

        # Store metrics for the current epoch
        best_fold_epoch_metrics['accuracy'].append(val_accuracy)
        best_fold_epoch_metrics['precision'].append(val_precision)
        best_fold_epoch_metrics['recall'].append(val_recall)
        best_fold_epoch_metrics['f1'].append(val_f1)
        best_fold_epoch_metrics['mrr'].append(val_mrr)
        best_fold_epoch_metrics['train_loss'].append(loss.item())
        best_fold_epoch_metrics['val_loss'].append(val_loss)
        best_fold_epoch_metrics['val_predictions'].append(val_predictions)
        best_fold_epoch_metrics['sorted_indices'].append(sorted_indices)

        # Early stopping based on validation loss
        if val_loss < best_fold_val_loss:
            best_fold_val_loss = val_loss
            best_fold_train_losses = train_losses.copy()
            best_fold_epoch_metrics['accuracy'].append(val_accuracy)
            best_fold_epoch_metrics['precision'].append(val_precision)
            best_fold_epoch_metrics['recall'].append(val_recall)
            best_fold_epoch_metrics['f1'].append(val_f1)
            best_fold_epoch_metrics['mrr'].append(val_mrr)
            best_fold_epoch_metrics['val_predictions'].append(val_predictions)
            best_fold_epoch_metrics['sorted_indices'].append(sorted_indices)
            
            patience_counter = 0
            
            if best_fold_val_loss < best_val_loss:
                best_val_loss = best_fold_val_loss
                best_model_state = model.state_dict()
                best_val_accuracies = val_accuracies.copy()
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
                break

    # Store metrics for the fold
    fold_accuracy_list.append(best_fold_epoch_metrics['best_accuracy'])
    fold_precision_list.append(best_fold_epoch_metrics['best_precision'])
    fold_recall_list.append(best_fold_epoch_metrics['best_recall'])
    fold_f1_list.append(best_fold_epoch_metrics['best_f1'])
    fold_mrr_list.append(best_fold_epoch_metrics['best_mrr'])

    # Store losses for the best model of the fold
    best_epoch_metrics['accuracy'].append(best_fold_epoch_metrics['accuracy'])
    best_epoch_metrics['precision'].append(best_fold_epoch_metrics['precision'])
    best_epoch_metrics['recall'].append(best_fold_epoch_metrics['recall'])
    best_epoch_metrics['f1'].append(best_fold_epoch_metrics['f1'])
    best_epoch_metrics['mrr'].append(best_fold_epoch_metrics['mrr'])
    best_epoch_metrics['train_loss'].append(best_fold_epoch_metrics['train_loss'])
    best_epoch_metrics['val_loss'].append(best_fold_epoch_metrics['val_loss'])

    print(f"Fold {fold + 1} - Accuracy: {fold_accuracy_list[-1]:.4f}, Precision: {fold_precision_list[-1]:.4f}, Recall: {fold_recall_list[-1]:.4f}, F1 Score: {fold_f1_list[-1]:.4f}")

# Print best model's evaluation metrics
print("\nBest Model Evaluation Metrics:")
print(f"Accuracy: {np.mean(fold_accuracy_list):.4f}")
print(f"Precision: {np.mean(fold_precision_list):.4f}")
print(f"Recall: {np.mean(fold_recall_list):.4f}")
print(f"F1 Score: {np.mean(fold_f1_list):.4f}")
print(f"MRR: {np.mean(fold_mrr_list):.4f}")

val_predictions = best_fold_epoch_metrics['val_predictions']
sorted_indices = best_fold_epoch_metrics['sorted_indices']

# Calculate average metrics across folds
avg_accuracy = np.mean(fold_accuracy_list)
avg_precision = np.mean(fold_precision_list)
avg_recall = np.mean(fold_recall_list)
avg_f1 = np.mean(fold_f1_list)
avg_mrr = np.mean(fold_mrr_list)

print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average MRR: {avg_mrr:.4f}")

# Save the best model
torch.save(best_model_state, f'/var/scratch/hwg580/{model_name}_best.pt')

# Plot
epoch_numbers = list(range(1, len(best_train_losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epoch_numbers, best_train_losses, label="Training Loss")
plt.plot(epoch_numbers, best_val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training and Validation Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(best_fold_epoch_metrics['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Precision
plt.subplot(2, 2, 2)
plt.plot(best_fold_epoch_metrics['precision'], label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(2, 2, 3)
plt.plot(best_fold_epoch_metrics['recall'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# F1 Score
plt.subplot(2, 2, 4)
plt.plot(best_fold_epoch_metrics['f1'], label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

def evaluate_model(predictions, true_values, sorted_indices, mask):

    true_values = true_values[mask].float()

    # print(true_values)
    # print(predictions)
    # print(true_values.size())
    # print(predictions.size())

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

metrics_dict = evaluate_model(val_predictions, labels, sorted_indices, val_mask)

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

# Load the best model for testing
model.load_state_dict(best_model_state)

# TESTING
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

# LOGGING
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

# # RDF
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








