# Annotating Scientific Text
## Setup
- `conda activate 11711`
- When not in use, `conda deactivate`


## Installations
- `conda install python=3.6`
- `pip3 install spacy==3.4` 
- `python3 -m spacy download en_core_web_lg`
- `conda install psycopg2`
- `pip install label-studio`


## Clone label studio
- In the annotations folder, `git clone https://github.com/HumanSignal/label-studio.git`
- `docker-compose up`
- Open Label Studio in your web browser at http://localhost:8080/ and create an account.

