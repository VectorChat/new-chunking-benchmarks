import argparse
import os
from PyPDF2 import PdfReader
from typing import Dict
import regex as re
from nltk.tokenize import sent_tokenize

def split_pdf_by_bookmarks(pdf_path: str) -> Dict[str, str]:
    reader = PdfReader(pdf_path)
    outlines = reader.outline
    page_content = [page.extract_text() for page in reader.pages]

    def get_bookmark_pages(bookmark, start_page=0):
        if isinstance(bookmark, list):
            return [get_bookmark_pages(item, start_page) for item in bookmark]
        else:
            title = bookmark['/Title']
            page_num = reader.get_destination_page_number(bookmark)
            return (title, page_num, start_page)

    flat_bookmarks = []
    def flatten_bookmarks(bookmarks, level=0):
        for item in bookmarks:
            if isinstance(item, list):
                flatten_bookmarks(item, level + 1)
            else:
                flat_bookmarks.append((item[0], item[1], item[2], level))

    bookmark_pages = get_bookmark_pages(outlines)
    flatten_bookmarks(bookmark_pages)
    flat_bookmarks.sort(key=lambda x: x[1])

    result = {}
    for i, (title, start, prev_start, level) in enumerate(flat_bookmarks):
        end = flat_bookmarks[i+1][1] if i+1 < len(flat_bookmarks) else len(page_content)
        content = ' '.join(page_content[start:end])
        result[title] = content

    return result

def clean_text(text: str) -> str:
  sentences = sent_tokenize(text)

  # Remove newlines within sentences
  cleaned_sentences = [sentence.replace('\n', ' ') for sentence in sentences]

  def clean_sent(sentence):
    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    # Remove leading and trailing spaces
    sentence = sentence.strip()
    # Remove spaces after ' and before ".,-’"
    sentence = re.sub(r'’ ', '', sentence)
    sentence = re.sub(r' ’', '', sentence)
    sentence = re.sub(r' \.', '', sentence)
    sentence = re.sub(r' \,', '', sentence)
    sentence = re.sub(r' \-', '', sentence)

    return sentence

  # for each sentence, clean it and join together
  cleaned_text = '\n'.join([clean_sent(sentence) for sentence in cleaned_sentences])
  return cleaned_text

argparser = argparse.ArgumentParser()
argparser.add_argument("--pdf_path", type=str, default="./dataset/the-fourth-wing/text.pdf")
argparser.add_argument("--ignore_cache", action="store_true")
argparser.add_argument("--output_dir", type=str, default="./dataset/the-fourth-wing/sections")

if __name__ == "__main__":
    args = argparser.parse_args()
    output_dir = args.output_dir
    pdf_path = args.pdf_path
    bookmark_text_dict = split_pdf_by_bookmarks(pdf_path)
            
    total_chars = 0
    
    for title, content in bookmark_text_dict.items():
        
        section_chars = len(content)
        
        total_chars += section_chars
        
        print(f"There are {section_chars} characters in {title}")
        
        path = f"{output_dir}/{title.replace(' ', '_')}.txt"                
        if os.path.exists(path) and not args.ignore_cache:
            print(f"Skipping {title} because it already exists")
            continue
        
        content = clean_text(content)
        
        with open(path, "w") as f:
            f.write(content)
        
        print(f"Saved {title} to {path}")
    
    print(f"Total characters: {total_chars}")