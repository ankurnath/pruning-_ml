from utils import *


from utils import *


problems = ['MaxCover','MaxCut','IM']

heading = {'MaxCover':'Maximum Cover','MaxCut':'Maximum Cut','IM':'Influence Maximization'}

for problem in problems:
    # print('Problem:',problem)
    print(f'& \\multicolumn{{6}}{{c|}}{{\\textbf{{{heading[problem]}}}}} \\\\ \\hline')

    datasets = ["Facebook",
                "Wiki",
                "Deezer",
                "Slashdot",
                "Twitter",
                "DBLP",
                "YouTube",
                'Skitter']
    
    for dataset in datasets:
        print(f'{dataset}', end='')

        df = load_from_pickle(file_path=f'{problem}_multibudget/data/{dataset}/MCTSPruner'
                                   ,quiet=True)
        
        ratios = df['Ratio'].values
        # print(' & ', end='')

        for ratio in ratios:
            print(f'& {round(ratio,4)} ', end='')
        print('\\\\')
    print(' \\hline')
        

        


                                   
                                   







