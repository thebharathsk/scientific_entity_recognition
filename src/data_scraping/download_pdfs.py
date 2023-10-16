import os
import hashlib
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def get_pdfs(input_path:str, output_path:str):
    """Download PDFs from metadata csv file.

    Args:
        input_path: path to input metadata csv file
        output_path: path to store output pdfs
    """
    #create output directory
    os.makedirs(output_path, exist_ok=True)
    
    #load urls from metadata
    metadata = pd.read_csv(input_path)
    urls = metadata['url'].tolist()
    
    #iterate through urls
    for url in tqdm(urls):
        #find hash of url uisng sha-256
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        #save path
        save_path = os.path.join(output_path, url_hash + '.pdf')
        
        #read url link and #download pdf
        try:
            #open url link
            response = requests.get(url+'.pdf')
            with open(save_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
        except:
            print("Time out for url: ", url)
            continue

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path to input metadata csv file", type=str)
    parser.add_argument("--output_path", help="path to store output pdfs", type=str)
    
    args = parser.parse_args()
    
    #download pdfs from metadata csv file
    get_pdfs(args.input_path, args.output_path)