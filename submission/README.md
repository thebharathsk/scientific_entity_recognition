To train the system, follow the steps below:

1. Install dependencies

```
conda create -n sciner python=3.10
conda activate sciner
pip install -r requirements.txt
```

2. Our model uses both manually annotated data and auto-annotated data(annotated using model trained on manually annotated data) for training. Download the data from the Google Drive [link](https://drive.google.com/file/d/15R1rzxYQiw0ohq57FUhTU3AcyEfMjdlk/view?usp=sharing). The code will take care splitting data into training and development sets. To use a smaller data for faster training, use the provided `train.conll` file instead

3. Run the command below to train the model
```
cd code
python train.py --data <path to downloaded data>  --model bert-large-cased  --exp_name training_exp  --num_epochs 5
```

4. Run the command below to test the model
```
cd code
python test.py --test_data <path to test data in csv format> --exp_path <path to experiments folder> --exp_name training_exp
```

5. At the end of this process, the results for test data will be stored in same folder i.e, `<path to experiments folder>/training_exp` as a csv file