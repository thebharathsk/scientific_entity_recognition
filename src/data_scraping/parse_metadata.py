import pandas as pd
import argparse
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def add_venue(df:pd.DataFrame):
    """Function to add venue column to dataframe.

    Args:
        df: dataframe containing paper metadata
    Returns:
        dataframe with venue column
    """
    
    #iterate over rows
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        #read url
        url = row['url']
        
        #open url link and read "venue" from it and ignore if time out
        try:
            response = requests.get(url)
            #parse html to identify links
            soup = BeautifulSoup(response.content, 'html.parser')
            hrefs = soup.find_all('a', href=lambda href: href and 'venues' in href)
                    
            #create a string of venues separated by |
            venues = "|"
            for venue_url in hrefs:
                venues+=venue_url.text + "|"
            
            #add venue column to dataframe
            df.loc[i, 'venues'] = str(venues)

        except:
            print("Time out for url: ", url)
            continue

    return df

def clean_value(value:str):
    """Function to clean value of a key-value pair.

    Args:
        value: value of a key-value pair
    Returns:
        cleaned value
    """
    #replace new lines with spaces
    value = value.replace('\n', ' ')
    #replace tabs with spaces
    value = value.replace('\t', ' ')
    #remove multiple consecutive spaces
    value = re.sub(' +', ' ', value)
    #remove any comma at the end of value
    value = value.rstrip(',')
    #remove any " character at first and last positions
    value = value.strip('"') 
    
    return value

def text_to_dict(text:list):
    """Function to convert list of text lines into a dictionary.

    Args:
        text: list of text lines for a paper
    Returns:
        dictionary of paper metadata
    """
    data = {}

    #pattern to identify start of key-value pair
    starting_pattern = r'^\s*([^=\s]+)\s+=\s+(.*)'
    
    key = ""
    value = ""
    for line in text:
        #check if line begins with '<key> = "' format
        match = re.match(starting_pattern, line)
        
        if match:
            if key != "" and value != "":
                #dump key and value into data
                value = clean_value(value)               
                data[key] = value
                
                key = ""
                value = ""

            #identify new key and value
            key = match.group(1)
            value = match.group(2)
                                    
        else:
            value += line
        
    #dump last key and value into data
    value = clean_value(value)               
    data[key] = value
    
    return data
    
def parse_bib(input_path:str, output_path:str):
    """Function to parse bib file and output csv file.

    Args:
        input_path: path to bib file
        output_path: path to output csv file
    """
    #find start and end lines for each paper
    start_idices = []
    end_indices = []
    lines = []
    
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('@'):
                start_idices.append(i)
            elif line.startswith('}'):
                end_indices.append(i)
            lines.append(line)

    #read data into a list of dictionaries
    data = []
    for i, (s_idx, e_idx) in enumerate(zip(start_idices, end_indices)):
            data.append(text_to_dict(lines[s_idx+1:e_idx]))
    
    #convert list of dictionaries into a dataframe
    df = pd.DataFrame(data)
        
    #add venue column
    df = add_venue(df)
    
    #save dataframe as csv file
    df.to_csv(output_path, index=True)
    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to bib file')
    parser.add_argument('--output_path', type=str, required=True, help='path to output csv file')
    
    args = parser.parse_args()
    
    #parse bib file
    parse_bib(args.input_path, args.output_path)