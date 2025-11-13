from itertools import product
import subprocess


search_space = dict(
input_chunk_lengths = [6, 12, 18],
hidden_sizes = [16, 32],
n_epochs = [20, 30, 50],
dropouts = [0.1, 0.3],
lstm_layers = [1, 2],
learning_rates = [2e-3, 5e-4]
)

for hp_vals in product(*search_space.values()):
    command = "papermill tft.ipynb ran_tft.ipynb"
    command += f" -p input_chunk_lengths {hp_vals[0]}"
    command += f" -p hidden_sizes {hp_vals[1]}"
    command += f" -p n_epochs {hp_vals[2]}"
    command += f" -p dropout {hp_vals[3]}"
    command += f" -p lstm_layers {hp_vals[4]}"
    command += f" -p learning_rate {hp_vals[5]}"

    print(command)
    subprocess.run(command.split())
    print("Success")
