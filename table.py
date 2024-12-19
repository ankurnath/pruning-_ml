from utils import *


problems = ['MaxCover','MaxCut','IM']

for problem in problems:
    folder = f'{problem}/data'
    datasets = os.listdir(folder)
    for dataset in datasets:
        file_path = os.path.join(folder,dataset,'MCTSPruner')
        data = load_from_pickle(file_path=file_path)
        print(f'Dataset: {dataset}')
        print(data[['Pruned Ground set(%)','Ratio(%)']])


# print(data)
# print(data.keys())
