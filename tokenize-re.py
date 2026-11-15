# The program for use Python's regular expression "re" library for tokenization
# basic tokenizer

import re

# following syntax to split a text on whitespace characters:
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

#Let’s modify the regular expression splits on whitespaces (\s), commas, and periods ([,.]):
text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
print(result)

# whitespace-free output
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
print(result)

# handle other types of punctuation, such as question marks, quotation marks, and the double-dashes
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)   