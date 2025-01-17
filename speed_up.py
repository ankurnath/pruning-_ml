import os
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_from_pickle  # Assuming `load_from_pickle` is defined here

# Specify problems
problems = ['MaxCover']

for problem in problems:
    folder = f'{problem}/data'
    datasets = os.listdir(folder)
    df = defaultdict(list)
    
    for dataset in datasets:
        dataset_folder = os.path.join(folder, dataset)
        algorithms = ['MCTSPruner', "GCOMB", 'LeNSE', 'COMBHelper']
        
        for algorithm in algorithms:
            try:
                data = load_from_pickle(os.path.join(dataset_folder, algorithm))
                df['Dataset'].append(dataset)
                df['Algorithm'].append(algorithm)
                df['Speed-up'].append(data['Speedup'].iloc[0])
            except:
                pass

    # Convert to pandas DataFrame
    df = pd.DataFrame(df)

    # Plotting
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 3))
    sns.barplot(data=df, x="Dataset", y="Speed-up", hue="Algorithm",edgecolor='black',
                palette=['#f9766e','#75ba75','#a20dfd','#f5945c'])
    sns.despine()
    
    # plt.title(f"Speed-up Comparison for {problem}", fontsize=14)
    # plt.xlabel("Dataset", fontsize=12)
    plt.xlabel('')
    plt.ylabel("Speed-up", fontsize=20)
    plt.legend(title="Algorithm", fontsize=10)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize =17)
    # Customize legend
    plt.legend(
        title="Algorithm", 
        fontsize=17, 
        title_fontsize=17,
        loc='upper right', 
        frameon=False
    )
    plt.locator_params(axis='y', nbins=8)
    plt.tight_layout()
    plt.savefig('speed_up.pdf',bbox_inches='tight', dpi=300)
    plt.show()
