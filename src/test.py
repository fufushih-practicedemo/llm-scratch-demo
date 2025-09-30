import urllib.request
import re
from SimpleTokenizer import SimpleTokenizerV1

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text= f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|==|\s)', raw_text)
all_words = sorted(set(preprocessed))

# create the vocal table
vocab = {token: integer for integer, token in enumerate(all_words)}

# tokenizer
tokenizer = SimpleTokenizerV1(vocab)
text = """
    "It's the last he painted, you know, " Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))