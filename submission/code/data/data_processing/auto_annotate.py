import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

#load utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import load_task_data, tokenize_data, get_word_ids, find_next_label_schematic

def test(args):
    """Test NER model

    Args:
        args: arguments for testing process
    """
    #list of labels
    label_list = ['O',
                'B-MethodName',
                'I-MethodName',
                'B-HyperparameterName',
                'I-HyperparameterName',
                'B-HyperparameterValue',
                'I-HyperparameterValue',
                'B-MetricName',
                'I-MetricName',
                'B-MetricValue',
                'I-MetricValue',
                'B-TaskName',
                'I-TaskName',
                'B-DatasetName',
                'I-DatasetName',
                'Ambiguous']
    
    #label to id
    label_to_id = {k: v for v, k in enumerate(label_list)}
    
    #load tokenizer
    model_path = os.path.join(args.exp_path, args.exp_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #load model
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    #load dataset
    test_data = load_task_data(args.input_path)
    test_dataloader = Dataset.from_pandas(pd.DataFrame({'tokens': test_data}))
    test_dataloader_tokenized = test_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': label_to_id})
    
    #hold outputs
    outputs = []
    
    #iterate over test data
    for i, d in tqdm(enumerate(test_dataloader_tokenized), total=len(test_dataloader_tokenized)):
        token_ids = torch.tensor(d['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(d['attention_mask']).unsqueeze(0)
        
        #predict labels
        predictions = model.forward(input_ids=token_ids, attention_mask=attention_mask)
        predictions_prob = torch.softmax(predictions.logits.squeeze(), dim=1)
        predictions = predictions.logits.squeeze()
        
        #convert predictions from tokens to word space
        #convert word ids and predictions to numpy array
        word_ids = np.array(get_word_ids(test_data[i:i+1], tokenizer))[0][1:-1]
        predictions = predictions.detach().cpu().numpy()[1:-1]
        predictions_prob = predictions_prob.detach().cpu().numpy()[1:-1]
        
        #iterate through word ids and predictions
        word_ids_unique = np.sort(np.unique(word_ids))
        
        #keep track of previous label
        prev_label = None
        for w_id in word_ids_unique:
            #find most common prediction for word id
            prediction_w_logits = predictions[word_ids == w_id]
            prediction_prob_w_id = predictions_prob[word_ids == w_id]
            prediction_w_id = find_next_label_schematic(prediction_w_logits, prev_label, label_to_id)

            collective_probability = np.prod(prediction_prob_w_id[:,prediction_w_id])
            
            if collective_probability > args.threshold:
                outputs.append(f'{test_data[i][w_id]} -X- _ {label_list[prediction_w_id]}\n')
                prev_label = prediction_w_id
            else:
                outputs.append(f'{test_data[i][w_id]} -X- _ {label_list[-1]}\n')
                prev_label = len(label_list) - 1
 
        #if input exceeds token number limit, add 'O' label to other words
        if  len(test_data[i]) > len(word_ids_unique):
            diff = len(test_data[i]) - len(word_ids_unique)
            for j in range(diff):
                outputs.append(f'{test_data[i][len(word_ids_unique)+j]} -X- _ {label_list[-1]}\n')
            
        #add empty target at the end of an input except for last input
        if i != len(test_dataloader_tokenized) - 1: 
            outputs.append('\n')
    
    #save outputs as conll file
    with open(os.path.join(args.output_path, f'auto_annotations_{args.exp_name}.conll'), 'w') as f:
        f.writelines(outputs)
    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to csv file defining data to be annotated')
    parser.add_argument('--output_path', type=str, required=True, help='path to annotations in CONNL format')
    parser.add_argument('--exp_path', type=str, required=False, default='../exps', help='path to save trained model')
    parser.add_argument('--exp_name', type=str, required=False, default='un-ner.model', help='name of experiment')
    parser.add_argument('--threshold', type=float, required=False, default=0.98, help='threshold for auto annotation')
    
    args = parser.parse_args()
    
    #test NER model
    test(args)