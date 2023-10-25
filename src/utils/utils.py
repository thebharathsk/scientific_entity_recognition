import math
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def load_labeled_conll_data(input_path:str):
    """Load data in CoNLL data and labels

    Args:
        input_path: path to input data
    Returns:
        Tuple of tokens and labels
    """
    #variable to hold data
    data = []
    
    #read data
    with open(input_path, 'r') as f:
        current_sample = []
        for i, line in enumerate(f):
            # #ignore 1st line
            # if i == 0:
            #     continue
            
            #if line is empty, add current sample to data
            if line == '\n':
                if len(current_sample) > 0:
                    data.append(current_sample)
                current_sample = []
            
            #else identify token and label
            else:
                line = line.strip()
                tags = line.split(' ')
                current_sample.append([tags[0], tags[-1]])            
    
    #add final sample
    if len(current_sample) > 0:
        data.append(current_sample)
    
    #seperate tokens and labels
    tokens = []
    labels = []
    for sample in data:
        current_tokens = []
        current_labels = []
        for t, l in sample:
            current_tokens.append(t)
            current_labels.append(l)
        tokens.append(current_tokens)
        labels.append(current_labels)
    
    return tokens, labels

def load_unlabeled_conll_data(input_path:str):
    """Load data in CoNLL data without any labels

    Args:
        input_path: path to input data
    Returns:
        data: data as a list
    """
    #load csv file
    df = pd.read_csv(input_path)
    
    #read "input" column into a list
    input_list = df["input"].tolist()

    data = []
    
    #read data
    #track current input
    current_sample = []
    for i, line in enumerate(input_list):
        # #ignore 1st line
        # if i == 0:
        #     continue
        
        #if line is empty, add current sample to data
        if isinstance(line, float) and math.isnan(line):
            if len(current_sample) > 0:
                data.append(current_sample)
            current_sample = []
        
        #else identify token
        else:
            current_sample.append(line)
    
    #add final sample
    if len(current_sample) > 0:
        data.append(current_sample)
    
    return data

def load_task_data(input_path:str):
    """Load data for annotation task

    Args:
        input_path: path to csv file defining annotation task 
    
    Returns:
        paragraphs to annotate
    """
    #load csv file
    task_data = pd.read_csv(input_path)
    
    #isolate rows where venues column contains "ANLP" or "NAACL" or "EMNLP"
    #and year column is or after 2022
    annotation_files = []
    for _, row in task_data.iterrows():
        if not isinstance(row['venues'], str):
            continue
        if row['year'] >= 2022 and ('|ANLP|' in row['venues'] or '|NAACL|' in row['venues'] or '|EMNLP|' in row['venues']):
            annotation_files.append(row['path'])
   
    #collect paragraphs from annotation files
    paragraphs = []
    for f in annotation_files:
        with open(f, 'r') as f:
            #add lines in file to paragraphs
            paragraphs.extend(f.readlines())
    
    #remove new line characters
    paragraphs = [p.strip() for p in paragraphs]
    
    #remove paragraphs that too short or too long
    paragraphs = [p for p in paragraphs if len(p.split(' ')) >= 10 and len(p.split(' ')) <= 400]
    
    #convert paragraphs into tokens
    tokens = [p.split(' ') for p in paragraphs]
        
    return tokens
 
def tokenize_data(examples:dict, tokenizer: AutoTokenizer, label_encoding_dict: dict):
    """Tokenize data

    Args:
        examples: examples to tokenize
        tokenizer: tokenizer to use
        label_encoding_dict: label to id mapping dictionary
        
    Returns:
        examples: tokenized examples
    """
    #tokenize inputs
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    
    #if data is unlabeled, return tokenized inputs
    if 'ner_tags' not in examples.keys():
        return tokenized_inputs
    
    #align labels with tokenized inputs
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        #keep track of label ids
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif label[word_idx] == 'Ambiguous':
                label_ids.append(-100)
            else :
                label_ids.append(label_encoding_dict[label[word_idx]])
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
   
def get_word_ids(examples:dict, tokenizer: AutoTokenizer):
    """Get word ids for a list of tokens

    Args:
        examples: examples to tokenize
        tokenizer: tokenizer to use
        
    Returns:
        word ids for each example post tokenization
    """
    #tokenize inputs
    tokenized_inputs = tokenizer(examples, truncation=True, is_split_into_words=True)
    
    #align labels with tokenized inputs
    word_ids = []
    for i in range(len(examples)):
        word_ids.append(tokenized_inputs.word_ids(batch_index=i))

    return word_ids