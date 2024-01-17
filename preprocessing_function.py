import tkinter as tk
from tkinter import filedialog
import pandas as pd
from io import StringIO
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
stop_words = set(stopwords.words('indonesian'))

def remove_stopwords(text):
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

  return " ".join(filtered_sentence)

def remove_punctuation(input_string):
  translator = str.maketrans('', '', string.punctuation)
  clean_string = input_string.translate(translator)

  return clean_string

def remove_numbers(text):
  result = ""
  for char in text:
      if not char.isnumeric():
          result += char

  return result


def remove_urls(text):
  url_pattern = r'https?://\S+'
  cleaned_text = re.sub(url_pattern, '', text)

  return cleaned_text

def remove_excess_space(text):
  cleaned_text = re.sub(r'\s+', ' ', text).strip()
  return cleaned_text

def clean_text(text):
  cleaned_text = remove_stopwords(text.lower())
  cleaned_text = remove_punctuation(cleaned_text)
  cleaned_text = remove_numbers(cleaned_text)
  cleaned_text = remove_excess_space(cleaned_text)