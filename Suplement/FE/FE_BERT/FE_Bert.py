from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from Suplement.model import GNN_graphpred
from torch_geometric.loader import DataLoader
import argparse


parser = argparse.ArgumentParser(description='About DDI-ESPredictor')
parser.add_argument('--dataset', type=str, default='drugbank',help='The dataset that the model is going to use for training the model')
parser.add_argument('--model', type=str, default= None, help='The deep forest model that used for predicition')
parser.add_argument('--outputfile', type=str, default= None, help='The file that the model is going to be saved at')
parser.add_argument('--input_drug1', type=str, default= None, help='The first drug that used for predicting the interaction')
parser.add_argument('--input_drug2', type=str, default= None, help='The Second drug that used for predicting the interaction')
parser.add_argument('--feature', type=str, default= "bert", help='feature extraction of drug pairs: bert or fingerprint')
args = parser.parse_args()


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ 
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    num_atom_features = 2 
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long) # Graph connectivity in COO format with shape [2, num_edges]
        edge_attr = torch.tensor(np.array(edge_features_list),dtype=torch.long) # Edge feature matrix with shape [num_edges, num_edge_features]
    else:  # no bond
        edge_index = torch.empty((2, 0), dtype=torch.long) 
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)#[0,num_bond_features]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def convert(args, model, device, loader):
    model.eval()
    node_representations = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            _, node_representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        graph_representation = torch.mean(node_representation, dim=0)
        node_representations.append(graph_representation.cpu().numpy())


    return node_representations 

def gnnsmile(s):
    rdkit_mol = AllChem.MolFromSmiles(s)
    data = mol_to_graph_data_obj_simple(rdkit_mol)
    data.id = torch.tensor(1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred(5, 300, 1,"last", 0.5, graph_pooling = 'mean', gnn_type = 'gin')# 5 GNN layers, 300 dimensionality of embeddings, 1 task, 0.5 dropout ratio
    model.from_pretrained('Suplement/FE/FE_BERT/model_gin/{}.pth'.format('Mole-BERT'))
    dataset = [data]
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.to(device)
    node_representations = convert(args, model, device, data_loader) 
    return(node_representations)
