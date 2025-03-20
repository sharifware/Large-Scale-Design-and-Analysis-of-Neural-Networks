#!/usr/bin/env python3
import sys
import torch

def main():
    # Ensure there is at least one input file and an output file.
    if len(sys.argv) < 3:
        print("Usage: {} input_file1.pt input_file2.pt ... output_file.pt".format(sys.argv[0]))
        sys.exit(1)
    
    # All arguments except the last are input files; the last is the output file.
    *input_files, output_file = sys.argv[1:]
    
    combined = {}
    counter = 0

    for file in input_files:
        data = torch.load(file)
        if isinstance(data, dict):
            # Iterate over all items in the dictionary.
            for key, value in data.items():
                # Create keys like 'network_1', 'network_2', etc.
                new_key = f"network_{counter+1}"
                combined[new_key] = value
                counter += 1
        else:
            print(f"Warning: {file} does not contain a dictionary; skipping.")

    torch.save(combined, output_file)
    print(f"Combined {counter} networks into '{output_file}'.")

if __name__ == '__main__':
    main()
