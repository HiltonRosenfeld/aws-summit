import requests
from bs4 import BeautifulSoup

import re
import unicodedata


def clean_text(text: str):
    # Normalize line breaks to \n\n (two new lines)
    text = text.replace("\r\n", "\n\n")
    text = text.replace("\r", "\n\n")

    # Replace two or more spaces with a single space
    text = re.sub(" {2,}", " ", text)

    # Remove leading spaces before removing trailing spaces
    text = re.sub("^[ \t]+", "", text, flags=re.MULTILINE)

    # Remove trailing spaces before removing empty lines
    text = re.sub("[ \t]+$", "", text, flags=re.MULTILINE)

    # Remove empty lines
    text = re.sub("^\s+", "", text, flags=re.MULTILINE)

    # remove unicode Non Breaking Space
    text = unicodedata.normalize('NFKC', text)

    return text


def website_to_txt(source: str):
    title = ""
    text = ""

    try:
        page = requests.get(source)
        if page.status_code != 200:
            error = f"Failed to retrieve the job posting at {source}. Status code: {page.status_code}"
            print(error)
            return "Error", error

        soup = BeautifulSoup(page.text, 'html.parser')
        title = soup.find('title')
        title = title.get_text().strip()
        body = soup.find('body')
        text = body.get_text()
        #text = clean_text(text)

    except Exception as e:
        error = f"Could not get the description from the URL: {source}\n{e}"
        return "Error", error

    return title, text

