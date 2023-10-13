import os
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

YEARS = [str(year) for year in range(2020, 2024)]  # The years for which we must download the PDFs
PDF_DEST_ROOT_PATH = './pdfs'  # Where the PDFs must be stored; format -> PDF_DEST_ROOT_PATH/<event_name>/<pdf_name>.pdf
BASE_URL = "https://aclanthology.org/"

response = requests.get(BASE_URL)
soup = BeautifulSoup(response.content, 'html.parser')
hrefs = soup.find_all('a', href=True)
events = [a['href'] for a in hrefs if '/events/' in a['href'] and any(year in a['href'] for year in YEARS)]


def download_pdf(pdf_url, pdf_file_path, index):
    response = requests.get(pdf_url)
    pdf_name = os.path.join(pdf_file_path, '{}_{}.pdf'.format(pdf_url.split("/")[-1][:-4], str(index).zfill(6)))
    with open(pdf_name, 'wb') as pdf_file:
        pdf_file.write(response.content)
    return pdf_name


def get_pdf_links(year_url):
    response = requests.get(year_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = [urljoin(BASE_URL, link['href']) for link in soup.select("a[href$='.pdf']")]
    return pdf_links


def scrape_pdfs_for_years():
    dataset_file_name = Path(os.path.join(PDF_DEST_ROOT_PATH, 'dataset.csv'))
    if dataset_file_name.is_file():
        data_list = pd.read_csv(dataset_file_name).values.tolist()
    else:
        data_list = []

    for event in events:
        print(f"Downloading event = {event}")
        event_year_url = urljoin(BASE_URL, event)
        pdf_links = get_pdf_links(event_year_url)
        event_year_name = event.split('/')[2]
        event_name = event_year_name.split('-')[0]
        year = event_year_name.split('-')[1]

        dest_folder_dir = os.path.join(PDF_DEST_ROOT_PATH, event_name, year, 'pdf')
        os.makedirs(dest_folder_dir, exist_ok=True)

        for i, pdf_url in enumerate(pdf_links):
            try:
                pdf_file_name = download_pdf(pdf_url, dest_folder_dir, i)
                data_list.append({'event': event, 'year': year, 'path': pdf_file_name})
                print(f"Downloaded: {pdf_url} @ {pdf_file_name}")
            except Exception as e:
                print(f"NOT Downloaded!: {pdf_url} due to {type(e).__name__}")
            break
        break

    dataset_df = pd.DataFrame(data_list)
    dataset_df.to_csv(dataset_file_name, index=False, mode='w+')
    print("Saved downloaded data record @ {}".format(dataset_file_name))


if __name__ == "__main__":
    scrape_pdfs_for_years()
