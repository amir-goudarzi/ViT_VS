import itertools
import os
import subprocess
from tqdm import tqdm
tqdm(disable=True)


# Hyperparameter ranges
mlp_sizes = [128, 256]
transformer_layers = [2, 5]
attention_heads = [2, 4]
embedding_dims = [64, 128]
learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
optimizers = ['Adam', 'SGD']
batch_sizes = [16, 64]

param_combinations = list(itertools.product(
    mlp_sizes,
    transformer_layers,
    attention_heads,
    embedding_dims,
    learning_rates,
    optimizers,
    batch_sizes
))

# Fixed parameters
patch_size = 16
positional_encoding = 'learned'
scheduler = "" # False
num_epochs = 500
attn_dropout = 0.0
mlp_dropout = 0.1
num_workers = 1


dir = "./"
dataset_path = "../../VisualSudokuData"
neptune_project = "GRAINS/visual-sudoku"
neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly\
9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMDQ2YThmNS1jNWU0LTQxZDItYTQxNy1lMGYzNTM4MmY5YTgifQ=="
split = 10

# File to track completed experiments
completed_experiments_file = "completed_experiments.txt"

# Load completed experiments
if os.path.exists(completed_experiments_file):
    with open(completed_experiments_file, "r") as file:
        completed_experiments = set(file.read().splitlines())
else:
    completed_experiments = set()

timeout_seconds = 720
# Loop over each combination
for params in param_combinations:

    mlp_size, num_transformer_layers, num_heads, embedding_dim, learning_rate, optimizer, batch_size = params


    # Create a unique identifier for the experiment
    experiment_id = f"{mlp_size}_{num_transformer_layers}_{num_heads}_{embedding_dim}_{learning_rate}_{optimizer}_{batch_size}"

    # Skip completed experiments
    if experiment_id in completed_experiments:
        continue


    # Construct command
    command = f"""
    python ./main.py \
      --patch_size "{patch_size}" \
      --positional_encoding "{positional_encoding}" \
      --embedding_dim "{embedding_dim}" \
      --num_transformer_layers "{num_transformer_layers}" \
      --mlp_dropout "{mlp_dropout}" \
      --attn_dropout "{attn_dropout}" \
      --mlp_size "{mlp_size}" \
      --num_heads "{num_heads}" \
      --batch_size "{batch_size}" \
      --optimizer "{optimizer}" \
      --learning_rate "{learning_rate}" \
      --scheduler "{scheduler}" \
      --num_epochs "{num_epochs}" \
      --num_workers "{num_workers}" \
      --dir "{dir}" \
      --dataset_path "{dataset_path}" \
      --neptune_project "{neptune_project}" \
      --neptune_api_token "{neptune_api_token}" \
      --split {split}
    """

    
    try:
        # Execute the command with a timeout
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)

        # Print output
        print(process.stdout)

        # Print errors (if any)
        if process.stderr:
            print("ERROR:", process.stderr)

        print("Process completed with exit code:", process.returncode)
        with open(completed_experiments_file, "a") as file:
            file.write(experiment_id + "\n")
            completed_experiments.add(experiment_id) 
    except subprocess.TimeoutExpired:
        print(f"Experiment timed out after {timeout_seconds} seconds.")
    