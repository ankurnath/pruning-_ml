import pandas as pd
from utils import *

df = pd.read_csv('baselines.csv')

data = {}
for i, problem in [(2, 'MaxCover'), (11, 'MaxCut'), (20, 'IM')]:
    data[problem] = {}
    for j in range(8):
        arr = df.iloc[i + j].values
        dataset = arr[0]  # Assuming the first column is the dataset name
        algorithms = ['QuickPrune', 'SS', 'GCOMB',  'COMBHelper', 'LeNSE', 'GNNPruner']
        data[problem][dataset] = {}
        
        # Adjust indexing to process each algorithm
        for idx, algorithm in enumerate(algorithms):
            k = 1 + idx * 3  # Start at 1, move in steps of 3 for Pr, Pg, C
            if k + 2 < len(arr):  # Ensure there are enough values to slice
                Pr, Pg, C = arr[k:k + 3]
                data[problem][dataset][algorithm] = {'Pr': Pr, 'Pg': Pg, 'C': C}


# print(data['MaxCover'])

for problem in ['MaxCover','MaxCut','IM']:
    folder = os.path.join(problem,'data')
    datasets = os.listdir(folder)
    for dataset in datasets:
        file_path = os.path.join(folder,dataset,'MCTSPruner')
        _data = load_from_pickle(file_path=file_path,quiet=True)
        Pg = (100-_data['Pruned Ground set(%)'].values[0])/100
        Pr = _data['Ratio(%)'].values[0]/100
        C = Pg*Pr
        # Pg = 1-pg
        # pr = _data['Ratio(%)'].values[0]/100
        Pr = round(Pr,4)
        Pg = round(Pg,4)
        C = round(C,4)

        # C = round(Pg*Pr,4)
        data[problem][dataset]['MCTSPruner'] = {'Pr': Pr, 'Pg': Pg, 'C': C}
        # print({'Pr': Pr, 'Pg': Pg, 'C': C})
        # print(_data['Ratio(%)'].values[0],_data['Pruned Ground set(%)'].values[0])


# print(data)

# Loop through the problems

heading = {'MaxCover':'Maximum Cover','MaxCut':'Maximum Cut','IM':'Influence Maximization'}
for problem in ['MaxCover', 'MaxCut', 'IM']:
    print('& \\multicolumn{12}{c|}{\\textbf{', end='')
    print(f'{heading[problem]}', end='')
    print('}} \\\\ \\hline')
    print()

    datasets = ['Facebook', 'Wiki', 'Deezer', 'Slashdot', 'Twitter', 'DBLP', 'YouTube', 'Skitter']
    algorithms = ['MCTSPruner', 'GCOMB', 'COMBHelper', 'LeNSE']

    # Loop through datasets
    for dataset in datasets:
        print('\\multicolumn{1}{|c||}{', end='')
        print(f'{dataset}', end='')
        print('}', end='')

        best_c = 0
        best_algo = None
        for algorithm in algorithms:

            try:
                if float(data[problem][dataset][algorithm]["C"]) > best_c:
                    best_c = float(data[problem][dataset][algorithm]["C"])
                    best_algo = algorithm
            except:
                pass
            # print(best_algo)



        # Loop through algorithms
        for algorithm in algorithms:
            # Print `Pr` value
            # print(f'& {data[problem][dataset][algorithm]["Pr"]}', end='')

            # # Print `Pg` value
            # print(f'& {data[problem][dataset][algorithm]["Pg"]}', end='')

            # # Print `C` value
            # print('& \\multicolumn{1}{c||}{', end='')
            # print(f'{data[problem][dataset][algorithm]["C"]}', end='')
            # print('}', end='')

            print(f'& {data[problem][dataset][algorithm]["Pr"]}', end='')

            # Print `Pg` value
            print(f'& {data[problem][dataset][algorithm]["Pg"]}', end='')

            # Print `C` value
            if algorithm == 'LeNSE':
                print('& \\multicolumn{1}{c|}{', end='')
            else:
                print('& \\multicolumn{1}{c||}{', end='')

            # value = data[problem][dataset][algorithm]["C"]
            # print(f'\\textbf{{{value}}}' if algorithm == best_algo else f'{value}', end='')

            if algorithm == best_algo:
                print('\\textbf{', end='')
                print(f'{data[problem][dataset][algorithm]["C"]}', end='')
                print('}', end='')
            else:
                print(f'{data[problem][dataset][algorithm]["C"]}', end='')
            print('}', end='')

        # End the row
        print('\\\\')

    # Print the horizontal line after each problem
    print('\\hline')


