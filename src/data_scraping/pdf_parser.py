import scipdf
import json

article_dict = scipdf.parse_pdf_to_dict('../../data/pdfs/aacl-2022/2022.aacl-main.1.pdf') # return dictionary
 

print(article_dict.keys())

#save as json
with open('../../data/test.json', 'w') as fp:
    json.dump(article_dict, fp)
