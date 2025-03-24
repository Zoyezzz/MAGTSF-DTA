# -*- coding:utf-8 -*-
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle
from  models.HMAGCN_ablation1 import *
#from models.LMANet import *
def train(model, device, train_loader,optimizer,ppi_adj,ppi_features,pro_graph,loss_fn,args,epoch):
    """
    Training function, which records the training-related logic.
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        optimizer.zero_grad()
        outputs= model(mol_data,pro_data,ppi_adj,ppi_features,pro_graph)
        output = outputs[0] # Only take the first element which is the prediction
        loss = loss_fn(output, mol_data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * args.batch,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def test(model, device, loader,ppi_adj,ppi_features,pro_graph):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_embeddings = torch.Tensor()
    total_drug_features = torch.Tensor()
    total_protein_features = None  # Initialize as None
    total_output_linear2 = torch.Tensor()

    print("Make prediction for {} samples".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            output= model(mol_data, pro_data, ppi_adj, ppi_features, pro_graph)

            batch_size = mol_data.y.size(0)  # Get the actual batch size
            output = output[:batch_size].cpu()
            total_preds = torch.cat((total_preds, output), 0) #predicted values
            total_labels = torch.cat((total_labels, mol_data.y[:batch_size].view(-1, 1).cpu()), 0) #ground truth
            total_embeddings = torch.cat((total_embeddings, embedding[:batch_size].cpu()), 0)
            total_drug_features = torch.cat((total_drug_features, drug_features[:batch_size].cpu()), 0)
            encoder_output = encoder_output.squeeze(0)[:batch_size].cpu()#??batch_size?????batch_size??????
            embedding_output = embedding_output[:batch_size].cpu()
            total_output_linear2 = torch.cat((total_output_linear2, embedding_output), 0)
            # ??????
            if total_protein_features is None:
                total_protein_features = encoder_output
            else:
                # print("Before cat:")
                # print(f"total_protein_features dtype: {total_protein_features.dtype}")
                # print(f"encoder_output dtype: {encoder_output.dtype}")
                # print(f"total_protein_features shape: {total_protein_features.shape}")
                # print(f"encoder_output shape: {encoder_output.shape}")
                # print(f"batch_size: {batch_size}")
                total_protein_features = torch.cat((total_protein_features, encoder_output), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_embeddings.numpy(), total_drug_features.numpy(), total_protein_features.numpy(),total_output_linear2.numpy()

def visualize_tsne(embeddings, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(filename)
    plt.close()

def main(args):
    dataset = args.dataset
    model_dict_ = {'HMANet': HMANet} # Two model architecture we proposed.
    modeling = model_dict_[args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(f'data/{dataset}/train.csv')# Reading training data.
    df_test = pd.read_csv(f'data/{dataset}/test.csv') # Reading test data.
    train_smile,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
    test_smile,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file) # Reading drug graph data from the serialized file.
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2) # Reading protein graph data from the serialized file.
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)# Reading PPI graph data from the serialized file.
    # 'ppi_index' is a dictionary that records the order of protein nodes in PPI, where the keys are protein sequences and the values are node indices

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)# Tensorization and sparsification of the adjacency matrix of the PPI graph.
    ppi_features = torch.Tensor(ppi_features).to(device)# Tensorization of the feature matrix of the PPI graph.
    pro_graph = proGraph(pro_data,ppi_index,device) # A function that encapsulates all protein graphs in the dataset into a single graph.

    train_dataset = DTADataset(train_smile, train_seq, train_label, mol_data = mol_data, ppi_index = ppi_index)
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data = mol_data, ppi_index = ppi_index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # training the model
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = f'results/{dataset}/'  + f'train_{model_st}.model'
    result_file_name = f'results/{dataset}/' + f'train_{model_st}.csv'
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, ppi_adj,ppi_features,pro_graph,loss_fn,args,epoch + 1)
        G, P = test(model, device, test_loader, ppi_adj,ppi_features,pro_graph)
        ret = [mse(G, P), concordance_index(G, P)]
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset,model_st)
        else:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset,model_st)

    # Create t-SNE visualizations
    labels = test_dataset.label  # Assuming labels are stored in test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'HMANet')
    parser.add_argument('--epochs', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'davis',choices = ['davis','kiba','Human'])
    parser.add_argument('--num_workers', type= int, default = 6)
    # parser.add_argument('--output', type=str, default='ppi_graph.pkl',help = 'The best performance of current model')
    args = parser.parse_args()
    main(args)