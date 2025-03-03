from utils import *

problems  = [
            'MaxCover',
            'MaxCut',
            'IM'
]
results = []  # List to store data before creating DataFrame
for problem in problems:

    folder = f'generelization/ER_200/{problem}/data'

    for dataset in [
                    'Facebook',
                    'Wiki',
                    'Deezer',
                    'Slashdot',
                    'Twitter',
                    'DBLP',
                    'YouTube',
                    'Skitter'
                    ]:
        for algorithm in [
                        'GNNPruner',
                        'MCTSPruner+GNNPruner+GuidedMCTS'
                        ]:
            
            file_path = os.path.join(folder,dataset,algorithm)

            _data = load_from_pickle(file_path=file_path,quiet=True)
            Pg = (100-_data['Pruned Ground set(%)'].values[0])/100
            Pr = _data['Ratio(%)'].values[0]/100
            C = Pg*Pr
            # Pg = 1-pg
            # pr = _data['Ratio(%)'].values[0]/100
            Pr = round(Pr,4)
            Pg = round(Pg,4)
            C = round(C,4)

            results.append({
                    "Problem": problem,
                    "Dataset": dataset,
                    "Algorithm": algorithm,
                    "Pg": Pg,
                    "Pr": Pr,
                    "C": C
                })

df = pd.DataFrame(results)

print(df)