import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data_distribution import dataset,base_dataset,MyDataset
from model import molecular_conv
from rdkit import Chem
from rdkit import DataStructs
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd


if __name__=='__main__':
    data = pd.read_excel('data_final.xlsx', index_col=0)

    train = data[data['train_set'] == 1]
    valid = data[data['valid_set'] == 1]
    test = data[data['test_set'] == 1]

    train_dataset = MyDataset(smiles=train['smi'], response=train['logkOH•'])
    valid_dataset = MyDataset(smiles=valid['smi'], response=valid['logkOH•'])
    test_dataset = MyDataset(smiles=test['smi'], response=test['logkOH•'])


    node_dim = base_dataset[0].num_node_features
    edge_dim = base_dataset[0].num_edge_features
    model = molecular_conv(out_channels=1,  # active or inactive
                           in_channels=node_dim, edge_dim=edge_dim,
                           hidden_channels=1024, num_layers=3, num_timesteps=8,
                           dropout=0.06)
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


    thresholds=np.arange(0,0.1,0.01)
    outputs_rmse=[]
    outputs_r2=[]
    for threshold in thresholds:
        train_fp=[]
        for i in train_dataset:
            Tanimoto=Chem.RDKFingerprint(i.mol)
            train_fp.append(Tanimoto)

        remained_item=[]
        for i in test_dataset:
            Tanimoto_test = Chem.RDKFingerprint(i.mol)
            temp=[]
            for train in train_fp:
                temp.append(DataStructs.FingerprintSimilarity(Tanimoto_test,train))
            mean_simu=sum(temp)/len(temp)


        # plt.figure(dpi=300)
        # plt.style.use('seaborn-whitegrid')
        #
        # plt.hist(remained_item, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        #
        # plt.title('Histogram of Tanimoto_index')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        #
        # plt.tight_layout()
        #
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        #
        # plt.axvline(np.mean(remained_item), color='orange', linestyle='dashed', linewidth=2, label='Mean')
        # plt.legend()
        #
        # plt.rc('font', size=12)
        #
        # plt.show()

            if mean_simu>threshold:
                remained_item.append(i)

        loader_test=DataLoader(remained_item,batch_size=len(remained_item))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion1=torch.nn.MSELoss(reduction='sum')

        model.to(device)

        model.load_state_dict(torch.load('best_val.model'))

        y_true, y_pred=predict(loader_test)
        r2=r2_score(y_true,y_pred)
        mse=np.sqrt(mean_squared_error(y_true,y_pred))
        outputs_rmse.append(mse)
        outputs_r2.append(r2)

        print(f'threshold:{threshold},RMSE:{mse},r2:{r2}')

        # for item in remained_item:
        #     smi = item.smiles
        #     condition = smi == data['smi']
        #     data.loc[condition, 'AD_screening'] = 1
        # data.to_excel('data_final.xlsx')

    # Set up a custom style for the plots
    plt.style.use('seaborn-whitegrid')

    # Create a figure with a defined size and resolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    # Plot the first graph (RMSE_test)
    ax1.plot(thresholds, outputs_rmse, marker='o', color='b', label='RMSE_test')
    ax1.set_xticks(thresholds)
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE_test', fontsize=12, fontweight='bold')
    ax1.set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
    ax1.legend()

    # Plot the second graph (R2_score)
    ax2.plot(thresholds, outputs_r2, marker='s', color='g', label='R2_score')
    ax2.set_xticks(thresholds)
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R2_score', fontsize=12, fontweight='bold')
    ax2.set_title('R2 Score', fontsize=14, fontweight='bold')
    ax2.legend()

    # Adjust the layout to avoid overlapping labels and titles
    plt.tight_layout()

    # Save the figure (optional)
    # plt.savefig('output_figure.png', dpi=300)

    # Show the plots
    plt.show()


























