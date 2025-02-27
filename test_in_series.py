import subprocess

# Define the commands to be run in parallel
problems = [
            'MaxCover', 
            'MaxCut',
            'IM'
            ]
datasets = [
            'Facebook', 
            'Wiki', 
            'Deezer', 
            'DBLP', 
            'Slashdot', 
            'Twitter', 
            'YouTube',
            'Skitter'
            ]

devices = {'MaxCover': 0, 'MaxCut': 0, 'IM': 0}

commands = []

for train_dist in [
                # 'ER',
                # 'BA',
                'ER_200'
                ]:
    for problem in problems:
        for dataset in datasets:
            # Construct the command for each combination of problem and dataset
            command = f'python test.py --problem {problem} --dataset {dataset} --device {devices[problem]} --train_dist {train_dist}'
            commands.append(command)  # Corrected: append the command to the list

# Start each command serially (one after the other)
for cmd in commands:
    subprocess.run(cmd, shell=True)

print("All commands have been completed.")
