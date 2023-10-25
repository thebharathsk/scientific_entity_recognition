import numpy as np
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#import utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import load_labeled_conll_data

#set matplotlib backend
matplotlib.use('Agg')

def analyze_training_data(input_path:str, output_path:str):
    """Analyze training data

    Args:
        input_path: path to training data
        output_path: path to save analysis results
    """
    #load data
    tokens, labels = load_labeled_conll_data(input_path)
    
    #dictionary to hold label counts
    label_counts = {}
    label_counts_fine = {}
    
    for _, labels_current in zip(tokens, labels):
        for label in labels_current:
            #store fine-grained label counts
            if label not in label_counts_fine:
                label_counts_fine[label] = 1
            else:
                label_counts_fine[label] += 1
            
            #store coarse-grained label counts
            if label.startswith('B-') or label.startswith('I-'):
                label = label[2:]
                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1
    
    #plot label counts
    plt.figure()
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.xlabel('labels')
    plt.ylabel('counts')
    plt.title('Label counts')
    
    #add colors to bars
    colors = list(mcolors.CSS4_COLORS.keys())
    colors = np.random.choice(colors, len(label_counts), replace=False)
    for i, bar in enumerate(plt.gca().patches):
        bar.set_color(colors[i])
        
    #save plot
    plt.savefig(os.path.join(output_path, 'label_counts.png'))
    
    #print label counts
    print('Total number of samples: {}'.format(len(labels)))
    print()
    print('Label counts:')
    for label, count in label_counts_fine.items():
        print('{}: {}'.format(label, count))
    
    #save fine grained label counts
    np.savez(os.path.join(output_path, 'label_counts.npz'), **label_counts_fine)
    
    #print a sample
    print()
    print('Sample:')
    idx = np.random.randint(0, len(labels))
    for token, label in zip(tokens[idx], labels[idx]):
        print('{}\t{}'.format(token, label))
                
if __name__ == "__main__":
    #parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, required=True, help='path to input data')
    argparser.add_argument('--output_path', type=str, required=True, help='path to save analysis results')
    args = argparser.parse_args()
    
    #analyze training data
    analyze_training_data(args.input_path, args.output_path)