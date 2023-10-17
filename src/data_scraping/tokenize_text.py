import json
import os
import argparse
import spacy

from tqdm import tqdm

def tokenize_data(input_path:str, output_path:str):
    """Tokenize text data

    Args:
        input_path: path to input txt files
        output_path : path to store output txt files containing tokenized data
    """
    #create output folder
    os.makedirs(output_path, exist_ok=True)
    
    #initialize tokenizer
    tokenizer = spacy.load("en_core_web_lg",
                           disable=['tok2vec', 'tagger', 
                                    'parser', 'attribute_ruler', 
                                    'lemmatizer', 'ner'])
    
    #list text files
    text_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.txt')]
    text_files.sort()
        
    #iterate through text files
    for text_file in tqdm(text_files):
        #open json file
        with open(text_file, 'r') as fp_input:
            #dump text to file
            save_file_name = os.path.basename(text_file)        
            with open(os.path.join(output_path, save_file_name), 'w') as fp_output:
                #parse each line in input file
                for line in fp_input:
                    #tokenize line
                    doc = tokenizer(line)
                    line_tokenized = ' '.join(token.text for token in doc)
                                        
                    #remove last space
                    line_tokenized = line_tokenized[:-2]+'\n'
                    
                    #write to file
                    fp_output.write(line_tokenized)
                    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, required=True, help="path to txt files")
    parser.add_argument("--output_path",  type=str, required=True, help="path to store output tokenized files")
    
    args = parser.parse_args()
    
    #tokenize text data
    tokenize_data(args.input_path, args.output_path)