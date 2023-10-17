import json
import os
import argparse
import spacy

from tqdm import tqdm

def compare_tokenizer(input_path:str):
    """Compare tokenizers of text data

    Args:
        input_path: path to input txt files
    """    
    #initialize tokenizer
    tokenizer_lg = spacy.load("en_core_web_lg",
                           disable=['tok2vec', 'tagger', 
                                    'parser', 'attribute_ruler', 
                                    'lemmatizer', 'ner'])
    tokenizer_trf = spacy.load("en_core_web_trf",
                           disable=['tok2vec', 'tagger', 
                                    'parser', 'attribute_ruler', 
                                    'lemmatizer', 'ner'])
    
    #list text files
    text_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.txt')]
    text_files.sort()
        
    #iterate through text files
    for text_file in tqdm(text_files):
        #open text file
        with open(text_file, 'r') as fp_input:
            #parse each line in input file
            for line in fp_input:
                #tokenize line
                doc_lg = tokenizer_lg(line)
                doc_trf = tokenizer_trf(line)
                line_tokenized_lg = ' '.join(token.text for token in doc_lg)
                line_tokenized_trf = ' '.join(token.text for token in doc_trf)
                
                if line_tokenized_lg != line_tokenized_trf:
                    print('mismatch')
                    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, required=True, help="path to txt files")
    
    args = parser.parse_args()
    
    #tokenize text data
    compare_tokenizer(args.input_path)