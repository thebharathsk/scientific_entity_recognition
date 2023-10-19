import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

def analyze_test_data(input_path:str, output_path:str):
    """Analysis of test data

    Args:
        input_path: path to input csv file
        output_path: path where analysis results are stored
    """
    #set seed
    np.random.seed(27011996)
    
    #open input file
    df = pd.read_csv(input_path)
    
    #read "input" column into a list
    input_list = df["input"].tolist()
    
    #analyze distribution of input lengths
    input_lengths = []
    split_indices = [i for i in range(len(input_list)) if isinstance(input_list[i], float) and math.isnan(input_list[i])]
    split_indices.insert(0, -1)
    split_indices.append(len(input_list))
    input_lengths = [split_indices[i+1] - split_indices[i] - 1 for i in range(len(split_indices)-1)]
    
    #plot histogram
    plt.hist(input_lengths, bins=100)
    plt.xlabel("input length")
    plt.ylabel("frequency")
    plt.title("Distribution of input lengths")
    plt.savefig(output_path + "/input_lengths.png")
    
    #print min, max, mean, median, std of input lengths
    print('Analysis of input lengths:')
    print('Total number of inputs: ', len(input_lengths))
    print("min: ", min(input_lengths))
    print("max: ", max(input_lengths))
    print("mean: ", np.mean(input_lengths))
    print("median: ", np.median(input_lengths))
    print("std: ", np.std(input_lengths))    

    #print 10 random inputs
    print('10 random inputs:')
    start_indices = np.random.randint(0, len(split_indices), size=10)
    end_indices = start_indices + 1
    for (s,e) in zip(start_indices, end_indices):
        s_ = split_indices[s]
        e_ = split_indices[e]
        print('Size = ', e_-s_+1)
        print(' '.join(input_list[s_+1:e_]))
        print('')
    
    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='input file path')
    parser.add_argument('--output_path', type=str, required=True, help='output file path')   
    args = parser.parse_args()
    
    #analyze test data
    analyze_test_data(args.input_path, args.output_path)
    
    
    