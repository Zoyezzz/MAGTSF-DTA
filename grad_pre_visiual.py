# -*- coding:utf-8 -*-
from models.HMAGC import *
from utils import *
import numpy as np
from torch_geometric.loader import DataLoader
import argparse
import torch


def grad_pre(device, loader, ppi_adj, ppi_features, proGraph, gradcam, test_seq):
    all_atom_att = []  
    all_labels = []  

    for i, data in enumerate(loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)

        _, atom_att = gradcam(mol_data, pro_data, ppi_adj, ppi_features, proGraph)

        atom_att_np = atom_att if isinstance(atom_att, np.ndarray) else atom_att.cpu().detach().numpy()
        all_atom_att.append(atom_att_np)  

        labels = np.array([i] * len(atom_att_np))  
        all_labels.append(labels)

        index = np.argwhere(atom_att_np > 0.92).reshape(-1)
        print("Index of atoms with attention > 0.92:", index)

        pocket_residues = []
        residue_att_values = {}  

        for j in index:
            residue = test_seq[i][j]
            att_value = atom_att_np[j].item()  
            pocket_residues.append(residue)  
            residue_att_values[residue] = att_value  
            print(f"Residue {residue} has attention {att_value}")

        pocket_string = ''.join(pocket_residues)
        print("Pocket String:", pocket_string)
        print("can ji ->atom_att:", residue_att_values) 

    all_atom_att = np.vstack(all_atom_att)  
    all_labels = np.concatenate(all_labels)   


def main(args):
    dataset = 'davis'
    model_dict_ = {'LMANet': LMANet, 'HMANet': HMANet}
    modeling = model_dict_[args.model]
    model_st = modeling.__name__

    path = f'results/{dataset}/train_{model_st}.model'
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    check_point = torch.load(path, map_location=device)
    model = modeling()

    from collections import OrderedDict
    new_checkpoint = OrderedDict()

    for key in check_point:
        if key.startswith('mol_mamba'):
            for layer_idx in range(3):
                new_key = key.replace('mol_mamba', f'mol_mamba_layers.{layer_idx}.mamba')
                new_checkpoint[new_key] = check_point[key].clone()
        elif key.startswith('mol_mamba_proj'):
            for layer_idx in range(3):
                new_key = key.replace('mol_mamba_proj', f'mol_mamba_layers.{layer_idx}.proj')
                new_checkpoint[new_key] = check_point[key].clone()
        elif key.startswith('mol_mamba_norm'):
            for layer_idx in range(3):
                new_key = key.replace('mol_mamba_norm', f'mol_mamba_layers.{layer_idx}.norm')
                new_checkpoint[new_key] = check_point[key].clone()
        else:
            new_checkpoint[key] = check_point[key]

    model.load_state_dict(new_checkpoint, strict=False)  
    
    model = model.to(device)

    test_smile = ['COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1']
    test_seq =['MARENGESSSSWKKQAEDIKKIFEFKETLGTGAFSEVVLAEEKATGKLFAVKCIPKKALKGKESSIENEIAVLRKIKHENIVALEDIYESPNHLYLVMQLVSGGELFDRIVEKGFYTEKDASTLIRQVLDAVYYLHRMGIVHRDLKPENLLYYSQDEESKIMISDFGLSKMEGKGDVMSTACGTPGYVAPEVLAQKPYSKAVDCWSIGVIAYILLCGYPPFYDENDSKLFEQILKAEYEFDSPYWDDISDSAKDFIRNLMEKDPNKRYTCEQAARHPWIAGDTALNKNIHESVSAQIRKNFAKSKWRQAFNATAVVRHMRKLHLGSSLDSSNASVSSSLSLASQKDCLAPSTLCSFISSSSGVSGVGAERRPRPTTVTAVHSGSK']
    test_label =[5.275723934173584,5.038913249969482]

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)
    pro_graph = proGraph(pro_data, ppi_index, device)

    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    gradcam = GradAAM(model, module=model.proGconv2)
    grad_pre(device, test_loader, ppi_adj, ppi_features, pro_graph,gradcam,test_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HMANet', help='0: LMANet 1:HMANet')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=6)
    args = parser.parse_args()
    main(args)
