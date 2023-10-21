import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from utils.utils import load_unlabeled_conll_data, tokenize_data, get_word_ids

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
                'I-DatasetName']
    
    #label to id
    label_to_id = {k: v for v, k in enumerate(label_list)}
    
    #load tokenizer
    model_path = os.path.join(args.exp_path, args.exp_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #load model
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    #load dataset
    test_data = load_unlabeled_conll_data(args.test_data)
    test_dataloader = Dataset.from_pandas(pd.DataFrame({'tokens': test_data}))
    test_dataloader_tokenized = test_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': label_to_id})
    
    #hold outputs
    outputs = []
    
    #iterate over test data
    count = 0
    for i, d in tqdm(enumerate(test_dataloader_tokenized), total=len(test_dataloader_tokenized)):
        token_ids = torch.tensor(d['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(d['attention_mask']).unsqueeze(0)
        
        #predict labels
        predictions = model.forward(input_ids=token_ids, attention_mask=attention_mask)
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        
        #convert predictions from tokens to word space
        #convert word ids and predictions to numpy array
        word_ids = np.array(get_word_ids(test_data[i:i+1], tokenizer))[0][1:-1]
        predictions = predictions.detach().cpu().numpy()[1:-1]
        
        #iterate through word ids and predictions
        word_ids_unique = np.unique(word_ids)
        for w_id in word_ids_unique:
            #find most common prediction for word id
            prediction_w_id = predictions[word_ids == w_id]
            prediction_w_id_mode = np.bincount(prediction_w_id).argmax()
            
            count += 1
            outputs.append({'id':count, 'target': label_list[prediction_w_id_mode]})
        
        #if input exceeds token number limit, add 'O' label to other words
        if len(word_ids) > len(test_data[i]):
            diff = len(word_ids) - len(test_data[i])
            for _ in range(diff):
                count += 1
                outputs.append({'id':count, 'target': label_list[0]})
            
        #add empty target at the end of an input except for last input
        if i != len(test_dataloader_tokenized) - 1: 
            count += 1
            outputs.append({'id':count, 'target': 'X'})
    
    #save outputs as csv file
    df = pd.DataFrame(outputs)
    df.to_csv(os.path.join(args.exp_path, args.exp_name, 'test_outputs.csv'), index=False)
    

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True, help='path to annotated data file for testing')
    parser.add_argument('--exp_path', type=str, required=False, default='../exps', help='path to save trained model')
    parser.add_argument('--exp_name', type=str, required=False, default='un-ner.model', help='name of experiment')
    
    args = parser.parse_args()
    
    #test NER model
    test(args)