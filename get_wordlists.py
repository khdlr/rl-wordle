import requests
import re
import numpy as np
from wordle_utils import encode

wordle_js = requests.get('https://www.powerlanguage.co.uk/wordle/main.e65ce0a5.js').text

# matches = re.match('var La=\[(?:"([a-z]+)",?)+\]', wordle_js)
def extract_wordlist(name):
    raw_list = re.findall(fr'{name}=\[([a-z",]+)\]', wordle_js)[0]
    return raw_list.replace('"', '').split(',')

solutions_str = extract_wordlist('La')
guesses_str   = extract_wordlist('Ta') + solutions_str

solutions = np.stack([encode(word) for word in solutions_str])
guesses   = np.stack([encode(word) for word in guesses_str])

np.savez_compressed('lists.npz', solutions=solutions, guesses=guesses)
