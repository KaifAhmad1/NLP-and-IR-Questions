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
