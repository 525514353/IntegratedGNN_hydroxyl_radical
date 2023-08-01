from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from tqdm.notebook import tqdm

from rdkit import Chem, DataStructs


import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from torch_geometric.loader import DataLoader
from model import molecular_conv

from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
#%% load data
data = pd.read_excel('data_final.xlsx', index_col=0)
data=data[data['outliers']!=1]

#%%
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
#%%
'''Splitting the dataset to train,valid,test'''
N = len(base_dataset)
M = N // 10

indices = np.random.permutation(range(N))

idx = {'train': indices[:8 * M],
       'valid': indices[8 * M:9 * M],
       'test': indices[9 * M:]}

modes = ['train', 'valid', 'test']

dataset = {m: Subset(base_dataset, idx[m]) for m in modes}
loader = {m: DataLoader(dataset[m], batch_size=200, shuffle=True) if m == 'train' else DataLoader(dataset[m], batch_size=200) for m in modes}

# '''Get train,valid,test dataset'''
# for train in dataset['train']:
#     smi = train.smiles
#     condition = smi == data['smi']
#     data.loc[condition, 'train_set'] = 1
#
# for valid in dataset['valid']:
#     smi = valid.smiles
#     condition = smi == data['smi']
#     data.loc[condition, 'valid_set'] = 1
#
# for test in dataset['test']:
#     smi = test.smiles
#     condition = smi == data['smi']
#     data.loc[condition, 'test_set'] = 1
#
# data.to_excel('data_final.xlsx')
#%% md
# attention fp
#%%
from torch_geometric.nn.models import AttentiveFP

node_dim = base_dataset[0].num_node_features
edge_dim = base_dataset[0].num_edge_features

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
model = molecular_conv(out_channels=1,  # active or inactive
                       in_channels=node_dim, edge_dim=edge_dim,
                       hidden_channels=1024, num_layers=3, num_timesteps=8,
                       dropout=0.06)

model=model.to(device)
#%%
train_epochs = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
#                                                 steps_per_epoch=len(loader['train']),
#                                                 epochs=train_epochs)
criterion1 = nn.MSELoss(reduction='sum')

#%%

def train(loader):
    total_loss = total_examples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()

        loss = criterion1(out, data.y)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
        total_examples += data.num_graphs

    return np.sqrt(total_loss / total_examples)


@torch.no_grad()
def tes(loader):
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
        loss = criterion1(out, data.y)
        total_loss += loss.item()
        total_examples += data.num_graphs
    return np.sqrt(total_loss / total_examples)


@torch.no_grad()
def predict(loader):
    y_pred = []
    y_true = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(out.cpu().numpy())

    return y_true, y_pred

#%%
best_val = float("inf")

learn_curve = defaultdict(list)
func = {'train': train, 'valid': tes, 'test': tes}

for epoch in tqdm(range(1, train_epochs+1)):
    loss = {}
    for mode in ['train', 'valid', 'test']:
        loss[mode] = func[mode](loader[mode])
        learn_curve[mode].append(loss[mode])

    # if loss['valid'] < best_val:
    #     torch.save(model.state_dict(), 'best_val.model')
    #     best_val = loss['valid']

    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d} Loss: ' + ' '.join(
            ['{} {:.6f}'.format(m, loss[m]) for m in modes]
        ))

fig, ax = plt.subplots()
for m in modes:
  ax.plot(learn_curve[m], label=m)
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_yscale('log')
plt.show()


model.load_state_dict(torch.load('best_val.model'))

for m in ['train','valid','test']:
    y_true, y_pred = predict(loader[m])
    print("{} {} {:.3f}".format(m, r2_score.__name__, r2_score(y_true, y_pred)))
    print("{} mse {:.3f}".format(m, tes(loader[m])))


