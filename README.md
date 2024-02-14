# NLP-and-IR-Coding Questions for Upcoming Interview: 
## NLP Interview Questions: 
### Implement a function to tokenize a given text.
``` Python 
def tokenization(text):
    tokens = text.split()
    return tokens

# Input: 
text = "Hello, how are you doing today?"
tokens = tokenization(text)
print(tokens)
```

### Write a program to remove stopwords from a text.
``` Python 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download NLTK resources like 
# punkt model for tokenization
nltk.download('stopwords')
nltk.download('punkt')
# Define a function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Input Parameters:
text = "This is an example sentence demonstrating how to remove stopwords from text."
text_without_stopwords = remove_stopwords(text)
print("Text with stopwords removed:")
print(text_without_stopwords)
```

### Write a program to calculate the frequency of each word in a text.
``` Python 
def calculate_word_frequency(text):
  words = text.split() # Split text into words 
  word_freq = {} # Empty Dictionary for storing word frequency 
  for word in words:
        # Update the frequency 
        word_freq[word] = word_freq.get(word, 0) + 1
  return word_freq

# Input 
text = "This is a simple example sentence. This sentence demonstrates word frequency calculation."
word_freq = calculate_word_frequency(text)
print("Word frequency:")
print(word_freq)
```

### Create a function to split a text into sentences.
``` Python 
import re 
def split_sentences(text):
  sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    # Split the text into sentences using the pattern,
    # and remove any empty strings from the resulting list.
  sentences = [sentence.strip() for sentence in re.split(sentence_endings, text) if sentence.strip()]
  return sentences

# Input 
text = "This is a sample text. It contains multiple sentences. How are you today?"
sentences = split_sentences(text)
print(sentences)
```
### Implement the bag of words model for a given corpus.
 A `bag of words` is a technique in natural language processing that represents text by counting the frequency of words, ignoring grammar and word order. It's used to convert text data into numerical feature vectors for machine learning tasks. Each document is represented as a vector where each element corresponds to the count of a particular word. While simple, it's effective for tasks like text classification and sentiment analysis.
``` Python 
# Bag of Words
from collections import Counter 
def preprocess_text(text):
  text = text.lower()
  tokens = text.split()
  return tokens 
def bag_of_words(corpus):
  words = [] # Empty list for storing Bag of Words 
  for document in corpus: 
    # Preprocess Documents 
    tokens = preprocess_text(document)
    # Extend words based on documents 
    words.extend(tokens)
    words_count = Counter(words)
  return words_count

# Input 
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]
result = bag_of_words(corpus)
print("Bag of Words Model:")
print(result)
```
### Explain and Implement a function to generate n-grams from a text.
The N-gram model is a technique in natural language processing (NLP) where we analyze text by considering sequences of N consecutive words. For instance, if N=2, we look at pairs of words (bigrams), and if N=3, we consider triplets (trigrams), and so forth. This allows us to capture the contextual relationships between words. We tokenize the text, breaking it down into individual words, and then slide a window of size N over the text to extract these sequences. Each sequence is called an N-gram. By counting how often each N-gram appears in a corpus of text, we can estimate the likelihood of encountering that sequence. This information is used in various NLP tasks such as language modelling, where we predict the next word in a sequence given the previous words.
``` Python 
def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i+n]))
    return ngrams
# Input:
text = "This is a sample text for generating n-grams."
n = 2 
result = generate_ngrams(text, n)
print(f"{n}-grams:")
print(result)
```
###  Implement text normalization techniques like case folding and accent removal.
``` Python 
import unicodedata
def normalize_text(text):
    normalized_chars = []
    for char in text:
        char_lower = char.lower()
        # Remove accents (diacritics)
        normalized_char = unicodedata.normalize('NFKD', char_lower)
        normalized_char = ''.join(c for c in normalized_char if not unicodedata.combining(c))
         # Append the normalized character to the list
        normalized_chars.append(normalized_char)
         # Join the normalized characters back into a string
    normalized_text = ''.join(normalized_chars)
    return normalized_text

text = "Héllo WÓRLD!"
normalized_text = normalize_text(text)
print(normalized_text) 
```
###  Write a spell checking program for a given text.
``` Python 
!pip install pyenchant
import enchant 
def spell_check(text):
  dictionary = enchant.Dict("en_US") # Dictionary Object for English Dic
  words = text.split()
  # Check mispelled words 
  misspelled_words = [word for word in words if not dictionary.check(word)]
  return misspelled_words

# Input 
text = 'Thiss iss a test to chekk the misspelled wordss.'
misspelled_words = spell_check(text)
if misspelled_words:
   print("Misspelled words found:")
   for word in misspelled_words:
    print(word)
else:
  print("No Misspelled words found!")
```
### Implement a program to segment a text into coherent parts.
``` Python 
# Using NLTK and punkt model 
import nltk
nltk.download('punkt')
def segment_text(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Print the segmented sentences
    for i, sentence in enumerate(sentences, 1):
        print(f"Part {i}: {sentence.strip()}\n")
# Input 
text = """
Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages. As such, NLP is related to the area of human–computer interaction. Many challenges in NLP involve natural language understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
"""
segment_text(text)
```
``` Python
# Simpler Approach without using NLTK and punkt
def segment_text_manually(text):
    parts = []
    print("Segment the text into coherent parts. Enter each part separately. Type 'done' when finished.\n")
    while True:
        part = input("Enter a coherent part: ").strip()
        if part.lower() == 'done':
            break
        parts.append(part)
    return parts

# Input
text = """
Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages. As such, NLP is related to the area of human–computer interaction. Many challenges in NLP involve natural language understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
"""
parts = segment_text_manually(text)
print("\nCoherent Parts:")
for i, part in enumerate(parts, 1):
    print(f"Part {i}: {part}\n")
```
