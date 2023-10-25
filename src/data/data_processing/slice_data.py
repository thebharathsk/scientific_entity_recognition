import os
import argparse
import pandas as pd
from tqdm import tqdm
import shutil

def is_correct_conference(venues:str):
    """Check if the conference is correct

    Args:
        venues: venues where paper is presented
    """
    #edge case of other datatype
    if not isinstance(venues, str):
        return False
    
    #check if it is a conference
    if "|ACL|" in venues or "|EMNLP|" in venues or "|NAACL|" in venues:
        return True
    else:
        return False

def split_data(input_file:str, 
               output_path:str, 
               token_path:str,
               num_manual:int):
    """Split data into multiple csv files

    Args:
        input_file: Input csv file
        output_path: Path where data slices are stored
        token_path: Path to tokenized data
        num_manual: Number of manually annotated data 
    """
    #create folders for each data slice
    ashwin_path = os.path.join(output_path, "manually_annotated", "ashwin")
    bharath_path = os.path.join(output_path, "manually_annotated", "bharath")
    odemuno_path = os.path.join(output_path, "manually_annotated", "odemuno")
    
    os.makedirs(os.path.join(ashwin_path, "input"), exist_ok=True)
    os.makedirs(os.path.join(ashwin_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(bharath_path, "input"), exist_ok=True)
    os.makedirs(os.path.join(bharath_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(odemuno_path, "input"), exist_ok=True)
    os.makedirs(os.path.join(odemuno_path, "annotations"), exist_ok=True)
    
    os.makedirs(os.path.join(output_path, "auto_annotated", 
                             "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "unannotated", 
                             "annotations"), exist_ok=True)
    
    #load csv file
    df = pd.read_csv(input_file)
    
    #shuffle rows in dataframe
    df = df.sample(frac=1, random_state=2701996).reset_index(drop=True)
    
    #create three data frames containing slices of original dataframe
    manually_annotated = []
    auto_annotated = []
    unannotated = []
    
    #keep track of number of manually annotated data
    manual_count = 0
    
    #iterate through rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        #get venues and year column data
        title = row["title"]
        venues = row["venues"]
        year = row["year"]
        path = os.path.join(token_path, row["hash"] + ".txt")
        
        #check if tokenized data exists
        if not os.path.exists(path):
            continue
        
        #create a dictionary with data
        data = {"title": title, "venues": venues, "year": year, "path": path}
        
        #check if it needs to be manually annotated
        if year >= 2022 and is_correct_conference(venues) and manual_count < num_manual:
            manually_annotated.append(data)
            manual_count += 1
        
        #check if it needs to be auto annotated
        elif year >= 2018:
            auto_annotated.append(data)
        
        #check if it needs to be unannotated
        else:
            unannotated.append(data)
    

    #print number of data in each slice
    print("Number of manually annotated data:", len(manually_annotated))
    print("Number of auto annotated data:", len(auto_annotated))
    print("Number of unannotated data:", len(unannotated))
    
    #create dataframes from lists
    manually_annotated = pd.DataFrame(manually_annotated)
    auto_annotated = pd.DataFrame(auto_annotated)
    unannotated = pd.DataFrame(unannotated)
    
    #split manually annotated data into three parts
    manually_annotated_ashwin = manually_annotated.iloc[:num_manual//3]
    manually_annotated_bharath = manually_annotated.iloc[num_manual//3:2*num_manual//3]
    manually_annotated_odemuno = manually_annotated.iloc[2*num_manual//3:]
    
    #save dataframes as csv files
    manually_annotated_ashwin.to_csv(os.path.join(output_path, "manually_annotated", 
                                                  "ashwin", "task.csv"), index=True)
    manually_annotated_bharath.to_csv(os.path.join(output_path, "manually_annotated", 
                                                   "bharath", "task.csv"), index=True)
    manually_annotated_odemuno.to_csv(os.path.join(output_path, "manually_annotated",
                                                    "odemuno", "task.csv"), index=True)
    auto_annotated.to_csv(os.path.join(output_path, "auto_annotated", "task.csv"),index=True)
    unannotated.to_csv(os.path.join(output_path, "unannotated", "task.csv"), index=True)
    
    #copy tokenized data to respective folders
    print('Copying input files to respective folders')
    for index, row in tqdm(manually_annotated_ashwin.iterrows(), total=manually_annotated_ashwin.shape[0]):
        shutil.copy(row["path"], os.path.join(ashwin_path, "input", os.path.basename(row["path"])))
    for index, row in tqdm(manually_annotated_bharath.iterrows(), total=manually_annotated_bharath.shape[0]):
        shutil.copy(row["path"], os.path.join(bharath_path, "input", os.path.basename(row["path"])))
    for index, row in tqdm(manually_annotated_odemuno.iterrows(), total=manually_annotated_odemuno.shape[0]):
        shutil.copy(row["path"], os.path.join(odemuno_path, "input", os.path.basename(row["path"])))
    
if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="path to input metadata csv file", type=str)
    parser.add_argument("--output_path", help="path where data slices are stored", type=str)
    parser.add_argument("--token_path", default='/home/bharathsk/acads/fall_2023/scientific_entity_recognition/data/extracted_tokens', help="path where tokenized data is stored", type=str)
    parser.add_argument("--num_manual", default=100, help="path where data slices are stored", type=int)
    
    args = parser.parse_args()
    
    #split data
    split_data(args.input_file, args.output_path, args.token_path, args.num_manual)