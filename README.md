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
###  Create a program to detect the language of a given text.
``` Python 
def detect_language_manually(text):
    # Define common words in different languages
    common_words = {
        'english': ['hello', 'how', 'are', 'you', 'good'],
        'french': ['bonjour', 'comment', 'ça', 'va', 'bien'],
        'spanish': ['hola', 'cómo', 'estás', 'bien', 'gracias'],
        # Add more languages and their common words here
    }
    # Count the occurrence of common words in the text
    word_count = {lang: sum(text.lower().count(word) for word in words) for lang, words in common_words.items()}
    # Identify the language with the highest word count
    detected_language = max(word_count, key=word_count.get)
    return detected_language

# Input 
text = """
Hola, ¿cómo estás? Me llamo Juan y vivo en España. 
"""
detected_language = detect_language_manually(text)
print(f"The detected language is: {detected_language}")
```
### Create a program for Extracting Name Entities including their labels using the Spacy library? 
``` Python 
import spacy
def ner(text):
  nlp = spacy.load('en_core_web_sm') 
  document = nlp(text)
  # Extract Name Entities: 
  entities = [(entity.text, entity.label_) for entity in document.ents]
  return entities 
# Input 
text = """
In a recent study conducted by Stanford University, researchers used Named Entity Recognition (NER) techniques to analyze news articles from various countries, including the United States, China, and Brazil. The study aimed to identify and classify named entities such as government officials, multinational corporations, geographic locations, and specific dates mentioned in the articles. The results revealed significant differences in the frequency and distribution of named entities across different regions, highlighting the importance of NER in cross-cultural information extraction tasks.
"""
entities = ner(text)
print("Named Entities:")
for entity, label in entities:
    print(f"{entity}: {label}")
```
###  Write a program to perform POS tagging on a text.
``` Python 
import nltk
from nltk.tokenize import word_tokenize
# Download the necessary NLTK models for POS Tagging punkt and averaged_perceptron_tagger
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(text):
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    return tagged_words
# Input
text = "The quick brown fox jumps over the lazy dog."
tagged_text = pos_tagging(text)
print("POS Tagging:")
print(tagged_text)
``` 
### Implement a stemming algorithm such as Porter or Snowball.
``` Python 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
porter_stemmer = PorterStemmer()

def porter_stemming(text):
    words = word_tokenize(text)
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# Input
text = "The quick brown foxes are jumping over the lazy dogs."
stemmed_text = porter_stemming(text)
print("Stemmed Text (Porter Algorithm):")
print(stemmed_text)
```
###  Create a function to perform lemmatization on a text.
``` Python 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text
# Inpur
text = "The quick brown foxes are jumping over the lazy dogs."
lemmatized_text = lemmatize_text(text)
print("Lemmatized Text:")
print(lemmatized_text)
```
### Explain Subword tokenization. Explain Byte Pair Encoding(BPE) and WordPiece and also the implementation of these algorithms. 
Subword tokenization is a text processing technique used in natural language processing (NLP) to break down words into smaller units called subwords. Unlike traditional tokenization methods that treat each word as a separate entity, subword tokenization operates more granularly, segmenting words into meaningful subunits or morphemes. This approach is beneficial for handling morphologically rich languages, and `out-of-vocabulary` words, and reducing the vocabulary size in NLP models.
1. **Byte Pair Encoding:**  Byte Pair Encoding (BPE) is a subword tokenization algorithm that iteratively merges the most frequent pairs of consecutive symbols until a target vocabulary size is reached, effectively capturing morphological variations and reducing vocabulary size for improved NLP model performance.

``` Python
 BPE Algorithm:
1. Initialize:
   - corpus = [list of texts]
   - vocabulary = set of unique characters in corpus
   - desired_vocab_size = desired size of vocabulary
   - splits = dictionary with each word split into characters
   - merges = empty dictionary

2. Learn Merges:
   while size of vocabulary < desired_vocab_size:
      a. Find most frequent pair:
         - Compute frequencies of character pairs in splits
         - Identify the pair with highest frequency
      b. Merge most frequent pair:
         - Update splits by merging occurrences of the pair
         - Add merged pair to merges
         - Update vocabulary

3. Tokenization:
   a. Break text into characters
   b. Apply learned merges to tokenize text

4. Return tokenized text
```

### Explain Dependency Parsing. Write a program to perform dependency parsing on a sentence.
Dependency parsing in NLP is about analyzing the grammatical structure of a sentence by identifying which words depend on others and how they're related. It represents these relationships as a directed graph, where each word is a node, and the links between them show dependency relations. This technique helps machines understand sentence structure and is crucial for tasks like part-of-speech tagging, named entity recognition, and machine translation.
``` Python 
import spacy 
def dependency_parsing(text):
  nlp = spacy.load('en_core_web_sm')
  document = nlp(text)
  for token in document:
        print(token.text, "-->", token.dep_, "-->", token.head.text)
# Input 
sentence = "The quick brown fox jumps over the lazy dog."
dependency_parsing(sentence)
```
### Explain Semantic role labeling. Write a program to perform semantic role labeling on a sentence.
Semantic Role Labeling (SRL) is a natural language processing task where words in a sentence are labeled with their semantic roles, such as agent, patient, or instrument. It helps computers understand the meaning of text by identifying who did what to whom. SRL is crucial for tasks like information extraction, question answering, and sentiment analysis.
``` Python
import spacy
def semantic_role_labeling_spacy(text):
  nlp = spacy.load('en_core_web_sm')
  document = nlp(text)
  for token in document:
    print(token.text, token.dep_, token.head.text)
# Input 
sentence = "The cat sat on the mat."
semantic_role_labeling_spacy(sentence)
```
### Explain Name Entity Linkage and Implement basic NEL System using suitable packages?
Named Entity Linking (NEL) is a process in natural language processing (NLP) that involves identifying specific named entities in text and connecting them to unique entries in a knowledge base. This helps disambiguate entities and allows systems to understand text more accurately. The process includes recognizing named entities, finding potential matches in a knowledge base, choosing the correct match based on context, and linking to the corresponding entry. NEL is essential for tasks like information retrieval, question answering, and knowledge graph construction.
``` Python 
!pip install wikipedia 
import spacy 
import wikipedia 
nlp = spacy.load("en_core_web_sm")

def entity_linking(text):
  document = nlp(text)

  for ent in document.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE']: 
            entity_name = ent.text
            try:
                page = wikipedia.page(entity_name)
                print(f"Entity: {entity_name} | Link: {page.url}")
            except wikipedia.exceptions.PageError:
                print(f"No Wikipedia page found for {entity_name}")
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation options for {entity_name}: {', '.join(e.options)}")

text = "Steve Jobs was the co-founder of Apple Inc. He was born in San Francisco."
entity_linking(text)
```
### Explain Name Entity Disambiguation and write a code for NED? 
Named Entity Disambiguation (NED) is the process of resolving ambiguities in named entities by determining their specific meanings in context. It involves recognizing named entities, analyzing surrounding context, and using external knowledge sources to identify the correct referents for ambiguous entities. NED is crucial for accurate understanding of natural language text and is used in various NLP tasks such as information retrieval and question answering.
``` Python 
knowledge_base = {
    "Steve Jobs": "Steven Paul Jobs was an American business magnate, industrial designer, and inventor.",
    "Apple Inc.": "Apple Inc. is an American multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services.",
    "Apple": "Apple is the fruit of the apple tree, which is a deciduous tree in the rose family best known for its sweet, pomaceous fruit, the apple."
}

def entity_disambiguation(entity_name):
    # Check if the entity is present in the knowledge base
    if entity_name in knowledge_base:
        return knowledge_base[entity_name]
    else:
        return "No description found for this entity."

entity_name = "Apple"
print("Description:", entity_disambiguation(entity_name))
```
### Explain Conference Resolution. Create a function to resolve coreferences in a text.
Coreference Resolution is a task in NLP that aims to identify and link together all mentions of the same entity within a text. For instance, in the sentence "Jane saw her reflection in the mirror," Coreference Resolution would recognize that "her" refers back to "Jane." By linking these mentions, the system can understand that they represent the same person. This task is crucial for various NLP applications, such as document summarization and question answering, where understanding the relationships between different mentions is essential for producing coherent interpretations of text. Coreference Resolution systems utilize linguistic features and contextual information to accurately identify coreferent expressions and generate cohesive interpretations of text.
``` Python 
import spacy
def resolve_coreferences(text):
    nlp = spacy.load("en_core_web_sm")
    document = nlp(text)
    resolved_text = text
    for ent in document.ents:
        if ent.root.head == ent.root:
            coref = ent.text
            for token in ent.root.children:
                if token.dep_ == 'poss':
                    coref += "'s"
                    break
            resolved_text = resolved_text.replace(ent.text, coref)
    return resolved_text

# Input 
text = "John went to his favorite restaurant. He ordered a burger."
resolved_text = resolve_coreferences(text)
print("Resolved Text:", resolved_text)
```
## Information Retrieval Interview Questions: 
### Explain the major components of the Information Retrieval System in detail. 

An Information Retrieval System consists of three main components:

### 1. Document Subsystem
The Document Subsystem deals with the management and organization of documents within the system. It comprises the following functions:
- 1. **Acquisition:** This involves the process of selecting and collecting documents from various sources. It may include web crawling, data scraping, or manual selection methods to gather relevant documents.
- 2. **Representation:** Once documents are acquired, they need to be structured and indexed for efficient retrieval. Representation involves tasks such as indexing, abstracting, and organizing documents based on their content, metadata, or other characteristics.
- 3. **File Organization:** Documents can be organized in different ways for storage and retrieval. This includes sequential or inverted indexing methods. Sequential organization stores documents in a linear fashion, while inverted indexing builds indexes of terms pointing to documents containing those terms, enabling faster searches.

### 2. User Subsystem
The User Subsystem focuses on understanding and catering to the needs of system users. It includes the following functionalities:
- 1. **Problem Identification:** This step involves recognizing the information needs and tasks of users. Understanding user requirements is crucial for delivering relevant search results.
- 2. **Query Representation:** Once the user's information needs are identified, the next step is to convert these needs into search queries. Query representation involves translating user concepts or requirements into formal query language constructs that the system can understand.
- 3. **Query Translation:** The translated queries need to be processed further to match system requirements and capabilities. Query translation involves converting user queries into system-specific query structures or commands that can be executed by the system to retrieve relevant information.
 
### 3. Searching / Retrieval Subsystem
The Searching / Retrieval Subsystem is responsible for executing search queries and retrieving relevant documents. It encompasses the following functions:
- 1. **Matching:** This involves comparing user queries with indexed documents to identify matches. Various matching algorithms and techniques are used to determine the relevance of documents to the user's query.
- 2. **Retrieved Documents:** Once relevant documents are identified, they are presented to the user based on search algorithms. This may involve ranking documents by relevance or other criteria to help users find the most useful information.
- 3. **Feedback:** User feedback is essential for refining search strategies and improving the retrieval process. Feedback mechanisms enable users to provide input on the relevance and usefulness of retrieved documents, which can be used to adjust search algorithms and improve future search results.

Overall, these components work together to facilitate efficient information retrieval, ensuring that users can find relevant documents quickly and easily.

### Commonly well-known components in IR Systems: 
1. **Reader:** 
The Reader component in an Information Retrieval (IR) system preprocesses documents through tasks like tokenization, lemmatization, and stemming. Tokenization breaks text into individual words or tokens for processing, while lemmatization reduces words to their base form, enhancing consistency. These tasks standardize text representation, reduce noise, and improve retrieval accuracy.
2. **Indexer:**
The Indexer component is responsible for creating and maintaining indexes of the documents stored in the document store. It processes each document, extracts relevant information, and builds an index structure that facilitates efficient and fast retrieval. This index typically includes terms, their frequencies, and pointers to the corresponding documents, enabling quick lookup during retrieval.
3.  **Retriever:**
The Retriever component in an Information Retrieval (IR) system retrieves relevant documents from the document store based on user queries. It utilizes methods like lexical search or dense retrieval with a bi-encoder.
Lexical search directly matches query terms against indexed terms in documents, while dense retrieval with a bi-encoder encodes both queries and documents into dense vectors and computes their similarity scores.
The method choice depends on factors like document collection size, query complexity, and the trade-off between retrieval speed and accuracy.
4. **Ranker:**
The Ranker component in an Information Retrieval (IR) system is pivotal for assessing the relevance of retrieved documents to a user query. Employing various scoring techniques, it encompasses traditional methods such as Vector Space Models like `TF-IDF`, probabilistic methods like `Okapi-BM25`, as well as cutting-edge approaches like `ELMo` and `BERT`. While TF-IDF prioritizes documents containing query terms with high frequency in the document but low frequency across the entire collection, advanced models like ELMo and BERT leverage extensive training data to learn intricate relevance patterns. This diversity in scoring techniques allows the Ranker to adapt to different retrieval scenarios and optimize relevance ranking, thereby enhancing the overall search experience for users.
5. **Reranker:** The Re-ranker, also known as a cross-encoder, is a component that further refines the ranking of retrieved documents. It addresses limitations of initial retrieval processes by considering more complex interactions between the query and documents. Unlike the initial ranker, which scores documents based on their individual relevance to the query, the Re-ranker considers the relevance of documents relative to each other. It may employ techniques like `pairwise comparison` or `learning-to-rank` algorithms to assign new scores to documents. The Re-ranker plays a crucial role in improving the relevance of search results by correcting mistakes made by the initial retrieval process and incorporating more context and semantics into the ranking.
6. **Query Processor:** 
The Query Processor component interprets user queries, parsing them into keywords and understanding their intent to generate relevant search results. It refines queries by adding synonyms or filtering out irrelevant terms to improve retrieval accuracy. By ensuring accurate query interpretation and effective translation into retrieval actions, the Query Processor plays a crucial role in optimizing search results and enhancing the overall user experience.
7. **Document Store / Knowledge Base:**
The Document Store houses all searchable documents, acting as the source for the Retriever component's document retrieval process based on user queries. The choice of storage mechanism—whether it's a database, file system, or specialized search engine like Elasticsearch or Solr—impacts the efficiency and scalability of retrieval operations. Each mechanism offers unique advantages and trade-offs in terms of indexing speed, query performance, and scalability. Databases provide structured data storage and efficient querying but may struggle with scalability for unstructured text. File systems offer fast access but lack advanced search features. Specialized search engines like Elasticsearch or Solr excel in full-text search, making them ideal for large document collections with complex retrieval needs.

### Term Frequency Calculation: Write a program to calculate the term frequency of terms in a document.
``` Python 
from collections import defaultdict
def calculate_tf(document):
    document = document.lower()
    terms = document.split()
    term_freq = defaultdict(int) # Empty Dictionary for tf 

    for term in terms:
        term_freq[term] += 1

    total_terms = len(terms)
    for term, frequency in term_freq.items():
        term_freq[term] = frequency / total_terms
    
    return term_freq
# Input 
document = "This is a sample document to demonstrate term frequency calculation. This document has repeated terms to illustrate term frequency."
tf = calculate_tf(document)
print("Term Frequency:")
for term, frequency in tf.items():
    print(f"{term}: {frequency}")
```
### Implement a function to calculate the document frequency of terms in a corpus.
``` Python 
from collections import defaultdict

def calculate_df(corpus):
    df = defaultdict(int)  # Empty dictionary for document frequency
    for document in corpus:
        # Create a set of unique terms in the current document
        unique_terms = set(document.split())
        # Update the document frequency f
        for term in unique_terms:
            df[term] += 1
    return df

# Input:
corpus = [
    "This is document 1. It contains some terms.",
    "Document 2 has different terms than document 1.",
    "Document 3 is another example document with some common terms.",
]
df = calculate_df(corpus)
print("Document Frequency:")
for term, frequency in df.items():
    print(f"{term}: {frequency}")
``` 
### Create an inverted index from a collection of documents.
An inverted index is a data structure used in information retrieval systems to efficiently map terms to the documents that contain them. Unlike a forward index, which maps documents to the terms they contain, an inverted index organizes data by terms, making it quicker to search for documents containing specific terms.
``` Python 
def create_inverted_index(documents):
    inverted_index = {}
    for doc_id, document in enumerate(documents):
        terms = document.split()  # Split document into terms
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = set()  # Initialize posting list for the term
            inverted_index[term].add(doc_id)  # Add document ID to posting list
    return inverted_index

# Input 
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

inverted_index = create_inverted_index(documents)
for term, posting_list in inverted_index.items():
    print(f"{term}: {posting_list}")
```
References: 
[Inverted Indexing for Text Retrieval](https://www.dcs.bbk.ac.uk/~dell/teaching/cc/book/ditp/ditp_ch4.pdf)
.

### 
