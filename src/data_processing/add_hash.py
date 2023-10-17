import pandas as pd
import hashlib
import argparse
from tqdm import tqdm

def add_hash_column(input_file:str):
    """Add hash column to csv file

    Args:
        input_file: Input file
    """
    #load csv file
    df = pd.read_csv(input_file)
    
    #iterate through rows and add hash column
    for index, row in tqdm(df.iterrows()):
        #get url
        url = row['url']
        
        #find hash of url uisng sha-256
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        #add hash column
        df.loc[index, 'hash'] = url_hash
        
    #save csv file
    df.to_csv(input_file, index=False)

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="path to input metadata csv file", type=str)
    
    args = parser.parse_args()
    
    #add hash column
    add_hash_column(args.input_file)
    