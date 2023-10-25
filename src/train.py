import os
import argparse
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import comet_ml
import pandas as pd
import numpy as np
from torch import nn
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollator, DataCollatorForTokenClassification
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from utils.utils import load_labeled_conll_data, tokenize_data

#global variables
#list of labels
LABEL_LIST = ['O',
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
LABEL_TO_ID = {k: v for v, k in enumerate(LABEL_LIST)}

def compute_metrics(p):
    """Compute metrics for NER task

    Args:
        p: predictions and labels

    Returns:
        dictionary of metrics
    """
    #define metric
    metric = load_metric("seqeval")
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    results_dict =  {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
        
    return results_dict


class SciNERTrainer(Trainer):
    def __init__(self, args,
                        model,
                        train_args, 
                        train_dataset, 
                        eval_dataset, 
                        data_collator, 
                        tokenizer, 
                        compute_metrics):
        super().__init__(model=model, 
                         args=train_args, 
                         data_collator=data_collator, 
                         train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, 
                         tokenizer=tokenizer, 
                         compute_metrics=compute_metrics)
        #beta hyperparameter
        beta=0.9999
        
        #load class counts
        class_counts = np.load(args.loss_weights, allow_pickle=True)
        
        class_weights_list = []
        for label in LABEL_LIST:
            class_weights_list.append(float(class_counts[label]))
            
        self.class_weights_t = torch.tensor(class_weights_list)
        
        #normalize class weights
        self.class_weights_t = (1 - beta) / (1 - beta ** self.class_weights_t)
        
        print('Weights')
        print(self.class_weights_t)
         
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_t.to(model.module.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train(args):
    """Train NER model
        Args:
            args: command line arguments
    """
    ##STEP-1: Load data
    #initialize logger
    if not args.no_log:
        comet_ml.init(project_name='anlp/hw2/')

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    #get labeled train/val and unlabeled test data
    all_data = load_labeled_conll_data(args.data)
    
    #split train/val data
    num_train = int(len(all_data[0]) * args.data_split)
    train_data = all_data[0][:num_train], all_data[1][:num_train]
    val_data = all_data[0][num_train:], all_data[1][num_train:]

    ##STEP-2: Tokenize data
    train_dataloader = Dataset.from_pandas(pd.DataFrame({'tokens': train_data[0], 'ner_tags': train_data[1]}))
    val_dataloader = Dataset.from_pandas(pd.DataFrame({'tokens': val_data[0], 'ner_tags': val_data[1]}))
    
    #tokenize data
    train_dataloader_tokenized = train_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': LABEL_TO_ID})
    
    val_dataloader_tokenized = val_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': LABEL_TO_ID})
    
    ##STEP-3: Train and evaluate
    #define model
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    
    #modify model
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_LIST))
    model.num_labels = len(LABEL_LIST)
    model.config.num_labels = len(LABEL_LIST)
    
    #define training arguments
    train_args = TrainingArguments(
        args.exp_path,
        evaluation_strategy = "epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=1e-5
    )

    #define trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    #define metric
    metric = load_metric("seqeval")
    
    #set training params
    trainer = SciNERTrainer(
        args,
        model,
        train_args,
        train_dataset=train_dataloader_tokenized,
        eval_dataset=val_dataloader_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #train model
    trainer.train()
    trainer.evaluate()
    trainer.save_model(os.path.join(args.exp_path, args.exp_name))
    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to annotated data file for training')
    parser.add_argument('--loss_weights', type=str, default='/home/bharathsk/acads/fall_2023/scientific_entity_recognition/results/training_data_analysis/label_counts.npz', required=False, help='path to dictionary class counts')
    parser.add_argument('--model', type=str, required=False, default='dslim/bert-large-NER', help='model name')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4, help='number of training epochs')
    parser.add_argument('--num_epochs', type=int, required=False, default=10, help='learning rate')
    parser.add_argument('--data_split', type=float, required=False, default=0.8, help='train/val split ratio')
    parser.add_argument('--exp_path', type=str, required=False, default='../exps', help='path to save trained model')
    parser.add_argument('--exp_name', type=str, required=False, default='un-ner.model', help='name of experiment')
    parser.add_argument('--no_log', action='store_true', help='log experiment to comet.ml')
    args = parser.parse_args()
    
    #train NER model
    train(args)