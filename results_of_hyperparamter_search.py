import pickle
import optuna
from matplotlib import pyplot as plt

if __name__=='__main__':
    with open('example-study.pkl', 'rb') as f:
        study = pickle.load(f)


    fig, ax = plt.subplots(figsize=(8, 6),dpi=300)
    ax.plot([t.value for t in study.trials], label='Objective Value', color='#1f77b4', linewidth=2)
    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Objective Value', fontsize=16)
    ax.legend(fontsize=16)


    ax.grid(linestyle='--', alpha=0.7)


    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(axis='x', colors='#444444')
    ax.tick_params(axis='y', colors='#444444')

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 6),dpi=300)
    optuna.visualization.matplotlib.plot_edf(study)
    plt.xlabel('Objective Value', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.title('Empirical Distribution Function', fontsize=16)


    plt.gca().set_facecolor('white')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    plt.gca().spines['left'].set_color('#444444')
    plt.gca().spines['bottom'].set_color('#444444')
    plt.gca().tick_params(axis='x', colors='#444444')
    plt.gca().tick_params(axis='y', colors='#444444')

    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 6),dpi=300)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Hyperparameter', fontsize=14)
    plt.title('Hyperparameter Importances', fontsize=16)


    plt.gca().set_facecolor('white')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    plt.gca().spines['left'].set_color('#444444')
    plt.gca().spines['bottom'].set_color('#444444')
    plt.gca().tick_params(axis='x', colors='#444444')
    plt.gca().tick_params(axis='y', colors='#444444')

    plt.tight_layout()
    plt.show()
