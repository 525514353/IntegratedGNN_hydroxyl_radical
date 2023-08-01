import torch
import torch.nn.functional as F
import numpy as np
from deepchem.feat import MolGraphConvFeaturizer
from scipy.sparse import coo_matrix
from torch import nn

from model import molecular_conv
import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from skimage.io import imread
from cairosvg import svg2png
import os
from torch_geometric.data import DataLoader
import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

df = pd.read_excel('data_final.xlsx')
df=df[df['AD_screening']==1]

def load_data():
    print('Loading data...')
    df = pd.read_excel('data_final.xlsx')
    df = df[df['AD_screening'] == 1]

    dataset=[]
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        mol = Chem.MolFromSmiles(row.smi)
        # Adjacency Matrix
        adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        mol = Chem.MolFromSmiles(row.smi)
        graph_data = featurizer._featurize(mol)
        pyg_data = graph_data.to_pyg_graph()
        pyg_data.mol = mol
        pyg_data.smiles = row.smi
        pyg_data.y=row['logkOH•']
        pyg_data.mol_num=row.num
        pyg_data.A=adj
        dataset.append(pyg_data)
    return dataset


def img_for_mol(mol, atom_weights=[]):
    # print(atom_weights)
    highlight_kwargs = {}
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = matplotlib.colormaps['bwr']
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors,
            'highlightBondColors':[]
        }
        # print(highlight_kwargs)

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(6)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
                        # highlightAtoms=list(range(len(atom_weights))),
                        # highlightBonds=[],
                        # highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp.png', dpi=600)
    img = imread('tmp.png')


    os.remove('tmp.png')

    return img


# mol=Chem.MolFromSmiles('CCCCC')
# img=img_for_mol(mol,atom_weights=[-0.8,-0.2,0.3,0.4,0.5])
# plt.imshow(img)
# plt.show()
def plot_explanations(model, data):
    mol_num = int(data.mol_num[0])
    # print(mol_num)
    row = df.loc[mol_num]
    # row=df[df['num']==mol_num]
    smiles = row.smi
    mol = Chem.MolFromSmiles(smiles)
    # breakpoint()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0][1].imshow(img_for_mol(mol))
    axes[0][1].set_title(row['Names'])


    axes[0][0].set_title('Adjacency Matrix')
    axes[0][0].imshow(data.A[0],cmap='Blues')


    axes[1][0].set_title('Grad-CAM')
    final_conv_acts = model.final_conv_acts.view(data.x.shape[0], 1024)
    final_conv_grads = model.final_conv_grads.view(data.x.shape[0], 1024)
    grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)[:mol.GetNumAtoms()]
    scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
    axes[1][0].imshow(img_for_mol(mol, atom_weights=scaled_grad_cam_weights))

    axes[1][1].set_title('UGrad-CAM')
    ugrad_cam_weights = ugrad_cam(mol, final_conv_acts, final_conv_grads)
    axes[1][1].imshow(img_for_mol(mol, atom_weights=ugrad_cam_weights))


    plt.savefig(f'explanations/{mol_num}_{row.Names}.png',dpi=600)
    plt.close('all')

def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map

def ugrad_cam(mol, final_conv_acts, final_conv_grads):
    # print('new_grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map[:mol.GetNumAtoms()]).reshape(-1, 1)
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(node_heat_map*(node_heat_map >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(node_heat_map*(node_heat_map < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map

if __name__=='__main__':
    criterion=nn.MSELoss()
    dataset = load_data()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = molecular_conv(out_channels=1,  # active or inactive
                           in_channels=30, edge_dim=11,
                           hidden_channels=1024, num_layers=3, num_timesteps=8,
                           dropout=0.06).to(device)

    model.load_state_dict(torch.load('best_val.model'))

    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)
    model.train()
    total_loss = 0
    for data in tqdm(loader):
        # breakpoint()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
        data.y=torch.tensor(data.y,dtype=torch.float32)
        loss = criterion(out, data.y)

        loss.backward()
        try:
            plot_explanations(model, data)
        # except ValueError as e:
        except Exception as e:
            print(e)
            continue
        # breakpoint()
        total_loss += loss.item() * data.num_graphs



