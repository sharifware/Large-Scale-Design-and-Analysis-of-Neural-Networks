import numpy as np
import pandas as pd
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Generate regression data')
parser.add_argument('--num_points', type=int, default=100000, 
                    help='Number of data points to generate (default: 100000)')
parser.add_argument('--output_file', type=str, default='./data/simpleReg.csv',
                    help='Path to output CSV file (default: ./data/simpleReg.csv)')
args = parser.parse_args()

np.random.seed(12)

# Generate data for the fn
num_training_points = args.num_points
a = np.random.randn(num_training_points, 1)
b = np.random.randn(num_training_points, 1)

print(a.min(), a.max())

# Define the target function, in future could make dynamic
def target_function(a, b):
    return (1/5) * a**2 - (1/10) * b**3

# Get labels
y = target_function(a, b)

data = pd.DataFrame({
    'a': a.flatten(),
    'b': b.flatten(),
    'y': y.flatten()
})

#for now, ok to upload to git but if the data gets too large we'll want to use env and store it locally
data.to_csv(args.output_file, index=False)
