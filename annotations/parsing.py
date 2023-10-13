import spacy 
  
def tokenize_raw_text(input_file_path, output_file_path):
    nlp = spacy.load('en_core_web_lg')

    # tokenize the input and save to a new file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for paragraph in input_file:
                doc = nlp(paragraph)
                tokenized_paragraph = ' '.join(token.text for token in doc)
                output_file.write(tokenized_paragraph)

    print("Tokenization completed. Output saved to", output_file_path)

if __name__ == '__main__':
    # tokenize raw input
    input_file_path = 'sample_data/input_bert_abstract.txt'
    output_file_path = 'sample_data/tokenized_output.txt'
    tokenize_raw_text(input_file_path, output_file_path)
