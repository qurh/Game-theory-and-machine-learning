import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

class DegreeAwareDynamicGNN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(DegreeAwareDynamicGNN, self).__init__()
        self.degree_encoder = nn.Linear(1, hidden_channels)
        
        # Increased capacity to process both node features and degree information
        self.conv1 = GCNConv(num_node_features + hidden_channels, hidden_channels * 2)
        self.conv2 = GCNConv(hidden_channels * 2, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        
        # Degree-aware attention mechanism
        self.degree_attention = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, param1, param2):
        # Calculate and encode node degrees
        node_degrees = degree(edge_index[0], x.size(0)).unsqueeze(1)
        normalized_degrees = node_degrees / node_degrees.max()
        degree_embedding = self.degree_encoder(normalized_degrees)
        
        # Concatenate node features with degree embedding
        x_augmented = torch.cat([x, degree_embedding], dim=1)
        
        # Apply the first parameter, now also considering degree
        degree_weight = torch.sigmoid(self.degree_attention(degree_embedding))
        x_augmented = x_augmented * (param1.unsqueeze(1) * degree_weight)
        
        # Graph Convolution layers with residual connections
        x1 = self.conv1(x_augmented, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # Apply the second parameter with degree awareness
        x1 = x1 * (param2.unsqueeze(1) * degree_weight)
        
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        
        # Residual connection
        x2 = x2 + x1[:, :hidden_channels]
        
        x3 = self.conv3(x2, edge_index)
        
        return F.log_softmax(x3, dim=1)

def simulate_over_time(model, adjacency_matrix, node_features, initial_labels, 
                      num_timesteps, param1_evolution, param2_evolution):
    edge_index = adjacency_matrix.nonzero().t().contiguous()
    
    # Calculate degree distribution statistics
    degrees = degree(edge_index[0], node_features.size(0))
    avg_degree = degrees.mean().item()
    max_degree = degrees.max().item()
    degree_std = degrees.std().item()
    
    label_ratios = []
    degree_label_correlation = []
    current_labels = initial_labels
    
    for t in range(num_timesteps):
        param1 = param1_evolution(t)
        param2 = param2_evolution(t)
        
        out = model(node_features, edge_index, param1, param2)
        current_labels = out.argmax(dim=1)
        
        # Calculate label ratios
        unique_labels, counts = torch.unique(current_labels, return_counts=True)
        ratios = counts.float() / len(current_labels)
        label_ratios.append(ratios)
        
        # Calculate correlation between node degrees and labels
        correlation = torch.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            label_mask = (current_labels == label)
            if label_mask.sum() > 0:
                label_degrees = degrees[label_mask]
                correlation[i] = (label_degrees.mean() - avg_degree) / max_degree
        degree_label_correlation.append(correlation)
    
    return label_ratios, degree_label_correlation, {
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'degree_std': degree_std
    }

def main():
    num_nodes = 100
    num_features = 10
    num_classes = 3
    num_timesteps = 50
    hidden_channels = 16
    
    # Create dummy data with more realistic degree distribution (preferential attachment)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    degrees = torch.ones(num_nodes)
    for i in range(2, num_nodes):
        # Connect to existing nodes with probability proportional to their degree
        probs = degrees[:i] / degrees[:i].sum()
        connections = torch.multinomial(probs, 2, replacement=False)
        for j in connections:
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1
    
    node_features = torch.randn(num_nodes, num_features)
    initial_labels = torch.randint(0, num_classes, (num_nodes,))
    
    def param1_evolution(t):
        return torch.ones(num_nodes) * (1 + 0.01 * t)
    
    def param2_evolution(t):
        return torch.ones(num_nodes) * torch.sin(torch.tensor(t * 0.1))
    
    model = DegreeAwareDynamicGNN(num_features, num_classes, hidden_channels)
    
    label_ratios, degree_correlations, degree_stats = simulate_over_time(
        model, adjacency_matrix, node_features, initial_labels,
        num_timesteps, param1_evolution, param2_evolution
    )
    
    print(f"Network statistics:")
    for key, value in degree_stats.items():
        print(f"{key}: {value:.2f}")
    print(f"Final label ratios: {label_ratios[-1]}")
    print(f"Final degree-label correlations: {degree_correlations[-1]}")

if __name__ == "__main__":
    main()