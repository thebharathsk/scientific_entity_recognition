import json
import os
import argparse

from tqdm import tqdm

def parse_json_data(input_path:str, output_path:str):
    """Parse jsons to extract text data

    Args:
        input_path: path to input json files
        output_path : path to store output text files
    """
    #create output folder
    os.makedirs(output_path, exist_ok=True)
    
    #list json files
    json_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.json')]
    json_files.sort()
    
    #iterate through json files
    for json_file in tqdm(json_files):
        #variable to store text
        text = []
        
        #open json file
        with open(json_file, 'r') as fp:
            paper_dict = json.load(fp)
            #add title to text
            text.append(paper_dict['title'])
            
            #add abstract
            text.append(paper_dict['abstract'])
            
            #parse sections
            sections = paper_dict['sections']
            for s in sections:
                #add section title to text
                text.append(s['heading'])
                
                #add section text to text
                section_tedxt = s['text']
                
                #split section text by \n and add to text
                text.extend(section_tedxt.split('\n'))
            
            #dump text to file
            save_file_name = os.path.basename(json_file).replace('.json', '.txt')
            with open(os.path.join(output_path, save_file_name), 'w') as fp:
                fp.write('\n'.join(text))

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, required=True, help="path to json files")
    parser.add_argument("--output_path",  type=str, required=True, help="path to store output text files")
    
    args = parser.parse_args()
    
    #parse jsons to extract text data
    parse_json_data(args.input_path, args.output_path)