import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool 
from dataclasses import dataclass
from typing import NamedTuple
from einops import rearrange, repeat
from torch_geometric.utils import dropout_adj
from mamba_ssm import Mamba
from torch.nn import MultiheadAttention
from mamba_ssm.models.mixer_seq_simple import MixerModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LMANet(nn.Module): # Bottom-Up strategy
    def __init__(self,n_output = 1,output_dim=128,num_features_xd = 78,num_features_pro = 33,num_cross_attn_heads=8):
        super(LMANet, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.num_cross_attn_heads = num_cross_attn_heads
        # Mamba layer for drug features
        # self.mol_mamba = Mamba(d_model=num_features_xd, d_state=32, d_conv=64)  
        # self.mol_mamba_proj = nn.Linear(num_features_xd, num_features_xd)
        # self.mol_mamba_norm = nn.LayerNorm(num_features_xd)
        # self.mol_mamba_dropout = nn.Dropout(0.1)
        
        self.mol_mamba_layers = nn.ModuleList([
            nn.ModuleDict({
              'mamba': Mamba(d_model=num_features_xd, d_state=32, d_conv=4),
              'proj': nn.Linear(num_features_xd, num_features_xd),
              'norm': nn.LayerNorm(num_features_xd),
              'dropout': nn.Dropout(0.1)
            }) for _ in range(3)
        ])

        #GCN encoder used for extracting drug features.
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # GCN encoder used for extracting protein features.
        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GCNConv(output_dim, output_dim)
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # GCN encoder used for extracting PPI features.
        self.ppiGconv1 = GCNConv(output_dim, 1024)
        self.ppiGconv2 = GCNConv(1024, output_dim)
        self.ppiFC1 = nn.Linear(output_dim,1024)
        self.ppiFC2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # Cross-Attention layer
        self.cross_attn = MultiheadAttention(output_dim, self.num_cross_attn_heads) 
        self.cross_attn_proj = nn.Linear(output_dim, output_dim) 
        self.cross_attn_norm = nn.LayerNorm(output_dim) 
        self.cross_attn_dropout = nn.Dropout(0.1)

        # classifier
        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self,mol_data,pro_data,ppi_edge,ppi_features,pro_graph):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        p_x,p_edge_index,p_edge_len,p_batch = pro_graph

        # Extracting drug features
        x_input = x
        x = x.unsqueeze(0) # Reshape for mamba 
        for layer in self.mol_mamba_layers:
           x_residual = x
           x = layer['mamba'](x)
           x = layer['dropout'](x)
           x = layer['proj'](x)
           x = layer ['norm'](x+x_residual)
        x = x.squeeze(0)

        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout2(self.relu(self.molFC1(x)))
        x = self.dropout2(self.molFC2(x))

        # Extracting protein structural features from protein graphs.
        p_x = self.bn1(self.relu(self.proGconv1(p_x, p_edge_index)))
        p_x = self.bn2(self.relu(self.proGconv2(p_x, p_edge_index)))
        p_x = self.bn3(self.relu(self.proGconv3(p_x, p_edge_index)))
        p_x = global_mean_pool(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        p_x = self.dropout2(self.proFC2(p_x))

        #DropEdge
        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.6, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training)
        # Extracting protein functional features from PPI graph.
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(p_x, ppi_edge))) # Using protein features extracted from the protein graph as the initial node features for the PPI graph.
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge)))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_x = self.dropout1(self.ppiFC2(ppi_x))
        ppi_x = ppi_x[seq_num] # Extracting the representations of the proteins corresponding to the current batch from the PPI node embeddings based on their indices.

        #combination
        x = x.unsqueeze(0) 
        ppi_x = ppi_x.unsqueeze(0) 
        xc, _ = self.cross_attn(x, ppi_x, ppi_x)  # Cross-attention, x is Query, p_x is Key and Value
        xc = xc.squeeze(0) 
        xc = self.cross_attn_proj(xc) 
        xc = self.cross_attn_dropout(xc)
        xc = self.cross_attn_norm(xc + x.squeeze(0)) 
        xc = self.cross_attn_norm(xc + ppi_x.squeeze(0))
        #xc = torch.cat((x, ppi_x), 1)
        # classifier
        xc = self.dropout1(self.relu(self.fc1(xc)))
        xc = self.dropout1(self.relu(self.fc2(xc)))
        embedding = xc
        out = self.out(xc)

        return out


class HMANet(nn.Module):

    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, num_features_pro=33, num_features_ppi=1442,deg=None,num_heads1=3,num_heads2=3, num_cross_attn_heads=8):
        super(HMANet, self).__init__()
        self.output_dim = output_dim  
        self.n_output = n_output 
        self.num_heads1 = num_heads1 
        self.num_heads2 = num_heads2 
        self.num_cross_attn_heads = num_cross_attn_heads 

        # Mamba layer for drug features
        self.mol_mamba = Mamba(d_model=num_features_xd*2, d_state=32, d_conv=4)  
        self.mol_mamba_proj = nn.Linear(num_features_xd*2, num_features_xd*2) 
        self.mol_mamba_norm = nn.LayerNorm(num_features_xd*2) 
        self.mol_mamba_dropout = nn.Dropout(0.1) 
        self.mol_mamba2 = Mamba(d_model=num_features_xd*4, d_state=32, d_conv=4)  
        self.mol_mamba_proj2 = nn.Linear(num_features_xd*4, num_features_xd*4) 
        self.mol_mamba_norm2 = nn.LayerNorm(num_features_xd*4) 
        self.mol_mamba_dropout2 = nn.Dropout(0.1)
        self.mol_mamba3 = Mamba(d_model=output_dim, d_state=32, d_conv=4)  
        self.mol_mamba_proj3 = nn.Linear(output_dim, output_dim) 
        self.mol_mamba_norm3 = nn.LayerNorm(output_dim) 
        self.mol_mamba_dropout3 = nn.Dropout(0.1)
        # self.mol_mamba_layers = nn.ModuleList([
        #     nn.ModuleDict({
        #       'mamba': Mamba(d_model=num_features_xd, d_state=32, d_conv=4),
        #       'proj': nn.Linear(num_features_xd, num_features_xd),
        #       'norm': nn.LayerNorm(num_features_xd),
        #       'dropout': nn.Dropout(0.1)
        #     }) for _ in range(3)
        # ])

        # GCN encoder used for extracting drug features.
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)  
        self.molGconv2 = GCNConv(num_features_xd * 2 , num_features_xd * 4) 
        self.molGconv3 = GCNConv(num_features_xd * 4 , output_dim)  #
        self.molFC1 = nn.Linear(output_dim, 1024) 
        self.molFC2 = nn.Linear(1024, output_dim) 

        # Mamba layer for protein features (commented out in the original code)
        # self.pro_mamba = Mamba(d_model=num_features_pro, d_state=16, d_conv=4) 
        # self.pro_mamba_proj = nn.Linear(num_features_pro, num_features_pro)
        # self.pro_mamba_norm = nn.LayerNorm(num_features_pro)
        # self.pro_mamba_dropout = nn.Dropout(0.1)

        # GCN encoder used for extracting protein features.
        self.proGconv1 = GCNConv(num_features_pro, 64) 
        self.proGconv2 = GCNConv(output_dim, output_dim)  
        self.proGconv3 = GCNConv(output_dim, output_dim) 
        self.proFC1 = nn.Linear(output_dim, 1024)   
        self.proFC2 = nn.Linear(1024, output_dim)  

        # GCN encoder used for extracting PPI features.
        self.ppiGconv1 = GCNConv(num_features_ppi, 1024) 
        self.ppiGconv2 = GCNConv(1024, output_dim)   
        self.ppiFC1 = nn.Linear(output_dim, 1024) 
        self.ppiFC2 = nn.Linear(1024, 64)   

        self.relu = nn.ReLU()  
        self.dropout1 = nn.Dropout(0.1) 
        self.dropout2 = nn.Dropout(0.2) 
        
        # Cross-Attention layer
        self.cross_attn = MultiheadAttention(output_dim, self.num_cross_attn_heads) 
        self.cross_attn_proj = nn.Linear(output_dim, output_dim) 
        self.cross_attn_norm = nn.LayerNorm(output_dim) 
        self.cross_attn_dropout = nn.Dropout(0.1) 
        
        # classifier
        self.fc1 = nn.Linear(output_dim, 1024)  
        self.fc2 = nn.Linear(1024, 512) 
        self.out = nn.Linear(512, self.n_output)  

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch   # Extract drug graph data
        seq_num = pro_data.seq_num # Extract the sequence number of the protein
        p_x, p_edge_index, p_edge_len, p_batch = pro_graph  # Extract protein graph data

        # Extracting drug features
        x_input = x
        #x = x.unsqueeze(0) 
        # for layer in self.mol_mamba_layers:
        #    x_residual = x
        #    x = layer['mamba'](x)
        #    x = layer['dropout'](x)
        #    x = layer['proj'](x)
        #    x = layer ['norm'](x+x_residual)
        # x = x.squeeze(0)  
        #print(x.shape)
        x = self.relu(self.molGconv1(x, edge_index))
        x_residual = x
        #print(x_residual.shape)
        x = x.unsqueeze(0)
        #print(x.shape)
        x = self.mol_mamba(x)
        x = self.mol_mamba_proj(x)
        x = self.mol_mamba_dropout(x)
        #print(x.shape)
        x = x.squeeze(0)
        x = self.mol_mamba_norm(x+x_residual)
        
        x = self.relu(self.molGconv2(x, edge_index)) 
        #print(x.shape)
        x_residual = x
        x = x.unsqueeze(0)
        x = self.mol_mamba2(x)
        x = self.mol_mamba_proj2(x)
        x = self.mol_mamba_dropout2(x)
        x = x.squeeze(0)
        x = self.mol_mamba_norm2(x+x_residual)
        #x = x.squeeze(0) 
        x = self.relu(self.molGconv3(x, edge_index))
        x_residual = x
        x = x.unsqueeze(0)
        x = self.mol_mamba3(x)
        x = self.mol_mamba_proj3(x)
        x = self.mol_mamba_dropout3(x)
        x = x.squeeze(0)
        x = self.mol_mamba_norm3(x+x_residual)
          
        #x = DrugFeatureExtractor(x,edge_index) 

        # Global mean pooling for drug features
        x = global_mean_pool(x, batch) 

        x = self.dropout2(self.relu(self.molFC1(x))) 
        x = self.dropout2(self.molFC2(x))  

        # DropEdge
        ppi_edge, _ = dropout_adj(edge_index=ppi_edge, p=0.2, force_undirected=True, num_nodes=max(seq_num) + 1,training=self.training) # Drop edges in PPI graph
        ppi_x = self.dropout1(self.relu(self.ppiGconv1(ppi_features, ppi_edge))) 
        ppi_x = self.dropout1(self.relu(self.ppiGconv2(ppi_x, ppi_edge))) 
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x))) 
        ppi_x = self.dropout1(self.ppiFC2(ppi_x)) 
        ppi_x = ppi_x[p_batch] 
        # ppi_x = ppi_x[seq_num] # another way to extract the protein features from ppi

        # Extracting protein structural features from protein graphs.
        p_x = self.relu(self.proGconv1(p_x, p_edge_index)) 
        p_x = torch.cat((torch.add(p_x, ppi_x), torch.sub(p_x, ppi_x)), -1) 
        p_x = self.relu(self.proGconv2(p_x, p_edge_index)) 
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))  
        p_x = global_mean_pool(p_x, p_batch)  
        p_x = self.dropout2(self.relu(self.proFC1(p_x))) 
        p_x = self.dropout2(self.proFC2(p_x))    
        p_x = p_x[seq_num] # Extract protein feature for current batch
        # print(x.shape)
        # print(p_x.shape)

        # combination
        # Cross-Attention
        x = x.unsqueeze(0) 
        p_x = p_x.unsqueeze(0) 
        xc, _ = self.cross_attn(x, p_x, p_x)  # Cross-attention, x is Query, p_x is Key and Value
        xc = xc.squeeze(0) 
        xc = self.cross_attn_proj(xc) 
        xc = self.cross_attn_dropout(xc) 
        xc = self.cross_attn_norm(xc + x.squeeze(0)) 
        xc = self.cross_attn_norm(xc + p_x.squeeze(0))
        #print(xc.shape)
        # classifier
        xc = self.dropout1(self.relu(self.fc1(xc))) 
        xc = self.dropout1(self.relu(self.fc2(xc)))  
        embedding = xc    # Store the embedding
        out = self.out(xc) # Output layer for classification

        return out # Return 