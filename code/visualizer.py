import numpy as np

# Load data from .npy file
file_path = '../data/training_dict.npy'  # Replace with the actual path to your file
training_dict = np.load(file_path, allow_pickle=True).item()

for key, values in training_dict.items():
    print(f"{key}: {values}")