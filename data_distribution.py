import numpy as np
import torch
from deepchem.feat import MolGraphConvFeaturizer
from matplotlib import pyplot as plt
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset, Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm
import seaborn as sns
data=pd.read_excel('data_final.xlsx')

def smi_to_pyg(smi, y):
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    mol=Chem.MolFromSmiles(smi)
    graph_data = featurizer._featurize(mol)
    pyg_data=graph_data.to_pyg_graph()
    pyg_data.y=torch.FloatTensor([y])
    pyg_data.mol=mol
    pyg_data.smiles=smi
    return pyg_data

#%%
class MyDataset(Dataset):
    def __init__(self, smiles, response):
        mols = [smi_to_pyg(smi, y) for smi, y in
                tqdm(zip(smiles, response), total=len(smiles))]
        self.X = [m for m in mols if m]

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)
#%%
base_dataset = MyDataset(data['smi'], data['logkOH•'])
# g = base_dataset[31]
# plot_mol_graph(g, figsize=(6,3), edge_label=True)
#%% Splitting the data set

N = len(base_dataset)
M = N // 10

indices = np.random.permutation(range(N))

idx = {'train': indices[:8 * M],
       'valid': indices[8 * M:9 * M],
       'test': indices[9 * M:]}

modes = ['train', 'valid', 'test']

dataset = {m: Subset(base_dataset, idx[m]) for m in modes}
loader = {m: DataLoader(dataset[m], batch_size=200, shuffle=True) if m == 'train' else DataLoader(dataset[m], batch_size=200) for m in modes}

def data_distribution():
    plt.rcParams["font.family"] = "Times new roman"
    data_1 = data[data['outliers'] != 1]
    x = data_1.iloc[:, 4]
    plt.figure(figsize=(8, 6), dpi=300)


    plt.hist(x=x, bins=60, color='skyblue', edgecolor='black')
    plt.xlabel('Value', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('Histogram of Values', fontsize=18, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    sns.set_context("paper")
    sns.set_style("whitegrid")


    plt.figure(figsize=(8, 6),dpi=300)
    sns.violinplot(x, inner='quart', linewidth=2, palette='pastel')


    plt.xlabel('proportion', fontsize=16, fontweight='bold')
    plt.ylabel('Value', fontsize=16, fontweight='bold')
    plt.title('Distribution of Values', fontsize=16, fontweight='bold')


    median_value = np.median(x)
    plt.axhline(y=median_value, color='red', linestyle='--', linewidth=2, label='Median')
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14)


    plt.tight_layout()


    # plt.savefig('violin_plot.png', dpi=300)


    plt.show()

if __name__=='__main__':
    data_distribution()
