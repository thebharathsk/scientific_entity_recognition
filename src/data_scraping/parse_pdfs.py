import scipdf
import json
import os
import argparse

from tqdm import tqdm

def parse_pdf_data(input_path:str, output_path:str):
    """Parse pdfs to json files

    Args:
        input_path: path to input pdf files
        output_path : path to store output json files
    """
    #create output folder
    os.makedirs(output_path, exist_ok=True)
    
    #list pdf files
    pdf_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.pdf')]
    pdf_files.sort()
    
    #iterate through pdf files
    for pdf_file in tqdm(pdf_files):
        try:
            #parse pdf to dictionary
            article_dict = scipdf.parse_pdf_to_dict(pdf_file)
            #create file name
            save_file_name = os.path.basename(pdf_file).replace('.pdf', '.json')
            
            #save as json
            with open(os.path.join(output_path, save_file_name), 'w') as fp:
                json.dump(article_dict, fp)    
        except:
            print("Error parsing pdf: ", pdf_file)
            continue        

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, required=True, help="path to pdf files")
    parser.add_argument("--output_path",  type=str, required=True, help="path to store output jsons")
    
    args = parser.parse_args()
    
    #parse pdfs into jsons
    parse_pdf_data(args.input_path, args.output_path)