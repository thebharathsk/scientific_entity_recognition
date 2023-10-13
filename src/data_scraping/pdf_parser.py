import scipdf
# article_dict = scipdf.parse_pdf_to_dict('../../data/pdfs/aacl-2022/2022.aacl-main.0.pdf') # return dictionary
 
# option to parse directly from URL to PDF, if as_list is set to True, output 'text' of parsed section will be in a list of paragraphs instead
article_dict = scipdf.parse_pdf_to_dict('https://www.biorxiv.org/content/biorxiv/early/2018/11/20/463760.full.pdf', as_list=False)

# # output example
# >> {
#     'title': 'Proceedings of Machine Learning for Healthcare',
#     'abstract': '...',
#     'sections': [
#         {'heading': '...', 'text': '...'},
#         {'heading': '...', 'text': '...'},
#         ...
#     ],
#     'references': [
#         {'title': '...', 'year': '...', 'journal': '...', 'author': '...'},
#         ...
#     ],
#     'figures': [
#         {'figure_label': '...', 'figure_type': '...', 'figure_id': '...', 'figure_caption': '...', 'figure_data': '...'},
#         ...
#     ],
#     'doi': '...'
# }

# xml = scipdf.parse_pdf('example_data/futoma2017improved.pdf', soup=True) # option to parse full XML from GROBID