import os
import argparse
import comet_ml
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from utils.utils import load_labeled_conll_data, tokenize_data

def train(args):
    """Train NER model

    Args:
        args: arguments for training process
    """
    ##STEP-1: Load data
    #initialize logger
    comet_ml.init(project_name='anlp/hw2/')
    
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
    train_dataloader_tokenized = train_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': label_to_id})
    
    val_dataloader_tokenized = val_dataloader.map(tokenize_data, batched=True, fn_kwargs={'tokenizer': tokenizer, 'label_encoding_dict': label_to_id})
    
    ##STEP-3: Train and evaluate
    #define model
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(label_list))

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
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        results_dict =  {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
            
        return results_dict
    
    #set training params
    trainer = Trainer(
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
    parser.add_argument('--model', type=str, required=False, default='dslim/bert-large-NER', help='model name')
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-4, help='number of training epochs')
    parser.add_argument('--num_epochs', type=int, required=False, default=3, help='learning rate')
    parser.add_argument('--data_split', type=float, required=False, default=0.8, help='train/val split ratio')
    parser.add_argument('--exp_path', type=str, required=False, default='../exps', help='path to save trained model')
    parser.add_argument('--exp_name', type=str, required=False, default='un-ner.model', help='name of experiment')
    
    args = parser.parse_args()
    
    #train NER model
    train(args)