import pandas as pd
import requests
import time
import sys
import os


DATA_PATH = "../data/"
ARCHIVE_PATH = DATA_PATH + "arxiv-metadata-oai-snapshot.json"
PDF_PATH = DATA_PATH + "pdfs/"

CATEGORY = sys.argv[1] if len(sys.argv) > 1 else "cs.AI"


def download_and_store_pdf(id):
    # check if pdf already exists. pdf file names are the same as the arxiv id
    if os.path.exists(PDF_PATH + f"{id}.pdf"):
        return
    url = f"https://export.arxiv.org/pdf/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(PDF_PATH + f"{id}.pdf", "wb") as f:
            f.write(response.content)
        time.sleep(1)
    else:
        print(f"Failed to download {id}")


if __name__ == "__main__":
    df = pd.read_json(ARCHIVE_PATH, lines=True)
    ids = df[df["categories"] == CATEGORY]["id"]

    max = len(ids)
    for id in ids:
        download_and_store_pdf(id)
        max -= 1
        print(f"{max} papers left to download", end="\r")
    print("Done")
