import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from model import molecular_conv
from data_distribution import MyDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_excel('data_final.xlsx',index_col=0)

train=data[data['train_set']==1]
valid=data[data['valid_set']==1]
test=data[data['test_set']==1]


train_dataset=MyDataset(smiles=train['smi'],response=train['logkOH•'])
valid_dataset=MyDataset(smiles=valid['smi'],response=valid['logkOH•'])
test_dataset=MyDataset(smiles=test['smi'],response=test['logkOH•'])

train_loader=DataLoader(train_dataset,batch_size=len(train_dataset),shuffle=False)
valid_loader=DataLoader(valid_dataset,batch_size=len(valid_dataset),shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False)

node_dim = train_dataset[0].num_node_features
edge_dim = train_dataset[0].num_edge_features
model = molecular_conv(out_channels=1,  # active or inactive
                       in_channels=node_dim, edge_dim=edge_dim,
                       hidden_channels=1024, num_layers=3, num_timesteps=8,
                       dropout=0.06)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
model.load_state_dict(torch.load('best_val.model'))

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

train_true,train_pre=predict(train_loader)
valid_true,valid_pre=predict(valid_loader)
test_true,test_pre=predict(test_loader)
plt.figure(figsize=(8,6),dpi=300)
plt.scatter(train_true,train_pre,label='train',color='#1f77b4', marker='o', s=50)
plt.scatter(valid_true,valid_pre,label='valid',color='#ff7f0e', marker='s', s=70)
plt.scatter(test_true,test_pre,label='test',color='#2ca02c', marker='^', s=60)
plt.legend(fontsize=16)
plt.xlabel('True',fontsize=16)
plt.ylabel('Predict',fontsize=16)


plt.gca().set_facecolor('white')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


plt.gca().spines['left'].set_color('#444444')
plt.gca().spines['bottom'].set_color('#444444')
plt.gca().tick_params(axis='x', colors='#444444')
plt.gca().tick_params(axis='y', colors='#444444')
plt.show()

print(np.sqrt(mean_squared_error(train_true,train_pre)))
print(r2_score(train_true,train_pre))

print(np.sqrt(mean_squared_error(valid_true,valid_pre)))
print(r2_score(valid_true,valid_pre))

print(np.sqrt(mean_squared_error(test_true,test_pre)))
print(r2_score(test_true,test_pre))
