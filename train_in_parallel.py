import subprocess

# Define the commands to be run in parallel
problems = [
            'MaxCover', 
            'MaxCut',
            ]
datasets = ['Facebook', 'Wiki', 'Deezer', 'DBLP', 'Slashdot', 'Twitter', 'YouTube', 'Skitter']

devices = {'MaxCover': 0, 'MaxCut': 0, 'IM': 0}

commands = []
for problem in problems:
    for dataset in datasets:
        # Construct the command for each combination of problem and dataset
        command = f'python main.py --problem {problem} --dataset {dataset} --device {devices[problem]}'
        commands.append(command)  # Corrected: append the command to the list

# Start each command in parallel without waiting for them to finish
for cmd in commands:
    subprocess.Popen(cmd, shell=True)

print("All commands have been started.")
