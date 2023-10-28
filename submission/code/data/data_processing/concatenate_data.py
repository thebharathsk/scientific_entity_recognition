file_1 = '/home/bharathsk/acads/fall_2023/scientific_entity_recognition/data/annotations/auto_annotated/annotations/auto_annotations_20231026_finetuning_stg2_lr5e-5_epochs20.conll'
file_2 = '/home/bharathsk/acads/fall_2023/scientific_entity_recognition/data/annotations/manually_annotated/all_annotations.conll'

data = [(file_1, 1), 
        (file_2, 15)]

new_data = []

for f, n in data:
    for i in range(n):
        with open(f, 'r') as ff:
            lines = ff.readlines()
            new_data.append(lines)

with open('/home/bharathsk/acads/fall_2023/scientific_entity_recognition/data/annotations/concatenated_annotations.conll', 'w') as f:
    for lines in new_data:
        for line in lines:
            f.write(line)