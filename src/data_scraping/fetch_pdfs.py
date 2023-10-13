import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = "https://aclanthology.org/"
response = requests.get(base_url)
soup = BeautifulSoup(response.content)
hrefs = soup.find_all('a', href=True)
years = [str(year) for year in range(2020, 2024)]
events = [a['href'] for a in hrefs if '/events/' in a['href'] and any(year in a['href'] for year in years)]
dest_root_path = './pdfs'


def download_pdf(pdf_url, folder_name):
    response = requests.get(pdf_url)
    pdf_name = os.path.join(folder_name, pdf_url.split("/")[-1])
    with open(pdf_name, 'wb') as pdf_file:
        pdf_file.write(response.content)


def get_pdf_links(year_url):
    response = requests.get(year_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = [urljoin(base_url, link['href']) for link in soup.select("a[href$='.pdf']")]
    return pdf_links


def scrape_pdfs_for_years(years):
    for year in years:
        for event in events:
            event_year_url = urljoin(base_url, event)
            pdf_links = get_pdf_links(event_year_url)
            dest_folder_dir = os.path.join(dest_root_path, event.split('/')[2])
            os.makedirs(dest_folder_dir, exist_ok=True)
            for pdf_url in pdf_links:
                download_pdf(pdf_url, dest_folder_dir)
                print(f"Downloaded: {pdf_url} @ {dest_folder_dir}")


if __name__ == "__main__":
    scrape_pdfs_for_years(years)
