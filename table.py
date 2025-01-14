from utils import *


problems = ['MaxCover','MaxCut','IM']

for problem in problems:
    # print('Problem:',problem)
    print('\multicolumn{3}{|c|}{',problem,'} \\\ \hline')
    folder = f'{problem}/data'
    datasets = os.listdir(folder)
    df = defaultdict(list)
    print('\multicolumn{1}{|l|}{Dataset}  & \multicolumn{1}{l|}{$P_g$} & \multicolumn{1}{l|}{$P_r$} \\\ \hline')
    for dataset in datasets:
        file_path = os.path.join(folder,dataset,'MCTSPruner')
        data = load_from_pickle(file_path=file_path,quiet=True)
        pg = data['Pruned Ground set(%)'].values[0]
        pg = round(pg,2)
        pr = data['Ratio(%)'].values[0]
        pr = round(pr,2)
        print(r'\multicolumn{1}{|c|}{%s} & \multicolumn{1}{c|}{%s} & %s \\ \hline' % (dataset,pg, pr))


    #     # print('\multicolumn{1}{|c|}{,problem,} & \multicolumn{1}{c|}{data['Pruned Ground set(%)']}  & {data['Ratio']}    \\\ \hline')
    #     df['Dataset'].append(dataset)
    #     df['Pruned Ground set(%)'].append(data['Pruned Ground set(%)'])
    #     df['Ratio(%)'].append(data['Ratio(%)'])
    #     # print(f'Dataset: {dataset}')
    #     # print(data[['Pruned Ground set(%)','Ratio(%)']])
    # df = pd.DataFrame(df)
    # print(df[['Dataset','Pruned Ground set(%)','Ratio(%)']].to_latex(index=False))
    # print(df[['Dataset','Pruned Ground set(%)','Ratio(%)']].to_latex(index=False))
    # break




# print(data)
# print(data.keys())
