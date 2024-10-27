import torch
import torch.optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from partner_switching_model import partner_switching_model
from ipdb import set_trace as st
from cProfile import Profile
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
from main import get_parameter_list_of_experiment_rho_vs_w_with_fixed_u, get_parameter_list_of_experiment_rho_vs_u_with_fixed_w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode ='train'
#new begin
def evaluate_model(model, parameter_list, u_interested=None, w_interested=None):
    results = []
    model.eval()
    with torch.no_grad():
        for idx, (n, m, u, w, rho, alpha, type_of_random_graph, _) in enumerate(parameter_list):
            if (u_interested is None and w_interested is None) or \
               (u_interested is not None and u == u_interested) or \
               (w_interested is not None and w == w_interested):
                try:
                    neighbor_list, strategy_list = generate_random_network(n, m, rho, type_of_random_graph)
                    edge_index, x = prepare_data_for_gcn_evaluate(neighbor_list, strategy_list)
                    edge_index, x = edge_index.to(device), x.to(device)
                    u, w = torch.tensor([u, w], dtype=torch.float).to(device)
                    y_pred = model(x, edge_index, u, w)
                    results.append((u.item(), w.item(), y_pred.item()))
                    if idx % 10 == 0:  # Print progress every 10 iterations
                        print(f"Processed {idx+1}/{len(parameter_list)} parameters")
                except Exception as e:
                    print(f"Error processing parameter set {idx}: {e}")
    
    print(f"Total results: {len(results)}")
    return results



# for evaluate

def prepare_data_for_gcn_evaluate(neighbor_list, strategy_list):
    edge_index = torch.tensor([[i, j] for i, neighbors in enumerate(neighbor_list) for j in neighbors], dtype=torch.long).t().contiguous()
    x = torch.tensor([[1 if s == 'C' else 0] for s in strategy_list], dtype=torch.float)
    return edge_index, x

#new end



def load_and_process_data():
    df = pd.read_csv("C:/Users/SAM/Documents/game theory and machine learning/codes for networks game/output.csv")
    data =[]
    # FIXME: .head(1000)
    for _, row in df.iterrows():
        data.append({
            "u": row[0],
            "w": row[1],
            "cooperative_ratio": row[2]
        })
    return data

def generate_random_network(n, m, rho, type_of_random_graph="ER"):
    """
    Generate a random network with given parameters.
    
    Args:`
    n (int): Number of nodes
    m (int): Number of edges
    initial_cooperative_ratio (float): Ratio of cooperative nodes
    type_of_random_graph (str): Type of random graph (default is "ER" for Erdős-Rényi)
    
    Returns:
    tuple: (neighbor_list, strategy_list)
    """
    if type_of_random_graph == "ER":
        edge_list = [(i,j) for i in range(n) for j in range(i+1, n)]
        edge_list = random.sample(edge_list, k=m)
        neighbor_list = [[] for _ in range(n)]
        for a, b in edge_list:
            neighbor_list[a].append(b)
            neighbor_list[b].append(a)
        
        strategy_list = ["C"] * int(n * rho) + \
                        ["D"] * (n - int(n * rho))
        random.shuffle(strategy_list)
    else:
        raise ValueError(f"Unknown random graph type: {type_of_random_graph}")
    
    return neighbor_list, strategy_list

#for train

def prepare_data_for_gcn(model):
    edge_index = torch.tensor([[i, j] for i, neighbors in enumerate(model._neighbor_list) for j in neighbors], dtype=torch.long).t().contiguous()
    x = torch.tensor([[1 if s == 'C' else 0] for s in model._strategy_list], dtype=torch.float)
    return edge_index, x

class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.6)
        self.lin = torch.nn.Linear(hidden_channels * 8 + 2, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, u, w):
        # GAT layers with dropout
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Global mean pooling (you can change it based on your need)
        x = x.mean(dim=0)

        # Concatenating u, w
        x = torch.cat([x, torch.tensor([u, w], dtype=torch.float).to(x.device)], dim=0)
        
        # Linear layer for final prediction
        x = self.lin(x)
        return x

class GCNModel_parameter_forward(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super().__init__()  
        # Adjust input feature size to account for u and w concatenation
        self.conv1 = GCNConv(num_node_features + 2, hidden_channels)  # +2 for u, w
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, u, w):
        # Concatenate u and w to the node features before GCN layers
        u_w_tensor = torch.tensor([u, w], dtype=torch.float).to(x.device)
        u_w_expanded = u_w_tensor.unsqueeze(0).repeat(x.size(0), 1)  # Repeat for all nodes
        x = torch.cat([x, u_w_expanded], dim=1)  # Concatenate along the feature dimension

        # Apply GCN layers with LeakyReLU activation
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)  # Optional dropout

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # Global mean pooling: averaging over all nodes
        x = x.mean(dim=0)

        # Final linear layer to predict the output
        x = self.lin(x)
        
        return x


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels + 2, 1)  # +3 for u, w
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, u, w):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        x = x.mean(dim=0)  # Global mean pooling
        x = torch.cat([x, torch.tensor([u, w], dtype=torch.float).to(device)], dim=0)
        x = self.lin(x)
        return x
    
class DenseModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(DenseModel, self).__init__()
        self.fc1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels + 2, 1)  # +2 for u, w

    def forward(self, x, edge_index=None, u=None, w=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        x = x.mean(dim=0)
        if u is not None and w is not None:
            x = torch.cat([x, torch.tensor([u, w], dtype=torch.float).to(x.device)], dim=0)
        
        x = self.lin(x)
        return x
    
class ImprovedGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(ImprovedGNNModel, self).__init__()
        self.param_nn = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.gat1 = GATConv(num_node_features + hidden_channels, hidden_channels, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * 8 + hidden_channels, hidden_channels, heads=8, dropout=0.6)
        self.lin = torch.nn.Linear(hidden_channels * 8 + hidden_channels, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, u, w):
        # Process parameters
        param_features = self.param_nn(torch.tensor([u, w], dtype=torch.float).to(x.device))
        param_features_expanded = param_features.unsqueeze(0).repeat(x.size(0), 1)
        
        # First GAT layer with parameter injection
        x = torch.cat([x, param_features_expanded], dim=1)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer with parameter injection (skip connection)
        x = torch.cat([x, param_features_expanded.repeat(1, 8)], dim=1)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Final prediction with parameter injection
        x = torch.cat([x.squeeze(), param_features], dim=0)
        x = self.lin(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Modified loss function
def parameter_aware_loss(pred, target, u, w, model, lambda_param=0.1):
    mse_loss = F.mse_loss(pred, target)
    param_loss = torch.mean(torch.abs(model.param_nn(torch.tensor([u, w], dtype=torch.float).to(pred.device))))
    return mse_loss - lambda_param * param_loss 

def main():
    print(f"Using device: {device}")

    # Create tensorboard writer
    log_dir = os.path.join("run", "experiment_1")
    writer = SummaryWriter(log_dir='./logs')


    # Load and process data
    data = load_and_process_data()
    
    N, M, rho, alpha = 1000, 5000, 0.5, 30
    
    print("Preparing dataset from processed data...")
    dataset = []
    for item in tqdm(data, desc="Preparing dataset"):
        neighbor_list, strategy_list = generate_random_network(N, M, rho)
        edge_index = torch.tensor([[i, j] for i, neighbors in enumerate(neighbor_list) for j in neighbors], dtype=torch.long).t().contiguous()
        x = torch.tensor([[1 if s == 'C' else 0] for s in strategy_list], dtype=torch.float)
        dataset.append((edge_index, x, item['u'], item['w'], item['cooperative_ratio']))

    print(f"Dataset size: {len(dataset)}")


    # Split dataset
    print("Splitting dataset into train and test sets...")
    random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    # Create and train model
    print("Creating and training model...")
    # model = GCNModel(num_node_features=1, hidden_channels=64).to(device)
    model =ImprovedGNNModel(num_node_features=1, hidden_channels=64).to(device)
    if mode == 'train':
        model.train()    
        # print(f"Model has parameter {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        criterion = torch.nn.HuberLoss()
        
    
        # Training loop
        best_test_loss = float('inf')
        for epoch in range(100):
            total_loss = 0
            for edge_index, x, u, w, y in tqdm(train_dataset, desc=f"Epoch {epoch+1}", leave=False):
                edge_index, x = edge_index.to(device), x.to(device)
                u, w = torch.tensor([u, w], dtype=torch.float).to(device)
                y = torch.tensor([y], dtype=torch.float).to(device)
                
                optimizer.zero_grad()
                out = model(x, edge_index, u, w)
                loss = parameter_aware_loss(out, y, u, w, model)
                # loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataset)
            
            # Evaluation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for edge_index, x, u, w, y in test_dataset:
                    edge_index, x = edge_index.to(device), x.to(device)
                    u, w = torch.tensor([u, w], dtype=torch.float).to(device)
                    y = torch.tensor([y], dtype=torch.float).to(device)
                    
                    out = model(x, edge_index, u, w)
                    test_loss += parameter_aware_loss(out, y, u, w, model)
                    # test_loss += criterion(out, y).item()
            avg_test_loss = test_loss / len(test_dataset)

            #log to tensorboard
            writer.add_scalar('Train Loss', avg_train_loss, epoch)
            writer.add_scalar('Test Loss', avg_test_loss, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
            
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
            

            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved with test loss: {best_test_loss / len(test_dataset):.6f}")

        print("Training completed.")
        writer.close()
    elif mode == "evalute the model":
        criterion = parameter_aware_loss(out, y, u, w, model)
        # criterion = torch.nn.HuberLoss()
        model.load_state_dict(torch.load('./best_model.pth'))
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for edge_index, x, u, w, y in test_dataset:
                edge_index, x = edge_index.to(device), x.to(device)
                u, w = torch.tensor([u, w], dtype=torch.float).to(device)
                y = torch.tensor([y], dtype=torch.float).to(device)
                
                out = model(x, edge_index, u, w)
                test_loss += criterion(out, y).item()

            print(f'hahaTest Loss: {test_loss / len(test_dataset):.6f}')

    
def evaluate_specific_parameter():
    u_list = [0, 0.01, 0.2, 0.8]
    w_list = [0, 0.05, 0.1, 0.5]
    model = ImprovedGNNModel(num_node_features=1, hidden_channels=64).to(device)

    # Check if the model is loaded correctly
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    parameter_list_of_experiment_rho_vs_w_with_fixed_u = get_parameter_list_of_experiment_rho_vs_w_with_fixed_u(u_list)
    parameter_list_of_experiment_rho_vs_u_with_fixed_w = get_parameter_list_of_experiment_rho_vs_u_with_fixed_w(w_list)

    print(f"Number of parameters for rho vs w: {len(parameter_list_of_experiment_rho_vs_w_with_fixed_u)}")
    print(f"Number of parameters for rho vs u: {len(parameter_list_of_experiment_rho_vs_u_with_fixed_w)}")

    gcn_results_rho_vs_w = evaluate_model(model, parameter_list_of_experiment_rho_vs_w_with_fixed_u)
    print(f"Results for rho vs w: {gcn_results_rho_vs_w[:5]}...")  # Print first 5 results
    gcn_results_rho_vs_u = evaluate_model(model, parameter_list_of_experiment_rho_vs_u_with_fixed_w)
    print(f"Results for rho vs u: {gcn_results_rho_vs_u[:5]}...")  # Print first 5 results
    # Before saving results, check if they're empty
    if not gcn_results_rho_vs_w or not gcn_results_rho_vs_u:
        print("Warning: Results are empty")
    else:
        print(f"Number of results for rho vs w: {len(gcn_results_rho_vs_w)}")
        print(f"Number of results for rho vs u: {len(gcn_results_rho_vs_u)}")

    # Save results to file
    with open('gcn_results_rho_vs_w.pkl', 'wb') as f:
        pickle.dump(gcn_results_rho_vs_w, f)
    with open('gcn_results_rho_vs_u.pkl', 'wb') as f:
        pickle.dump(gcn_results_rho_vs_u, f)

    print("Evaluation completed and results saved.")

if __name__ == "__main__":
    main()
    evaluate_specific_parameter()
    