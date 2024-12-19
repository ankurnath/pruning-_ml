import subprocess


# Define the commands to be run in parallel
commands = [
    "python main.py --problem MaxCover --dataset Facebook --device 0",
    # "python main.py --problem MaxCover --dataset Wiki --device 0",
    "python main.py --problem MaxCut --dataset Facebook --device 0",
    # "python main.py --problem MaxCut --dataset Wiki --device 0",
]

# Start each command in parallel without waiting for them to finish
for cmd in commands:
    subprocess.Popen(cmd, shell=True)

print("All commands have been started.")