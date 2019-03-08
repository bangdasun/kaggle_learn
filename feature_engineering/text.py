
import re
import numpy as np
import pandas as pd


def text_len(x):
    """ Extract string length """
    try:
        return len(str(x))
    except:
        return 0


def count_word(x, sep=None):
    """ Extract number of words in a string """
    try:
        return len(str(x).split(sep))
    except:
        return 0


def count_unique_word(x, sep=None):
    """ Extract number of unique words in a string """
    try:
        return len(set(str(x).split(sep)))
    except:
        return 0


def count_symbol(x, symbol=None):
    """ Extract number of symbol in a string """
    try:
        return str(x).count(symbol)
    except:
        return 0
		
def count_capital_letters(x):
    """ Extract number of captial letters in a string """
    try:
        return sum([s.isupper() for s in list(str(x))])
    except:
        return 0

def search_symbol(x, symbol=None):
    """ Search symbol and return first match place """
    result = re.search(symbol, str(x))
    try:
        return result.start()
    except:
        return -1
