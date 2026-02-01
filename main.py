import nltk
import stanza
import re
import string
# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stanza.download('en')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize Stanza pipeline
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
text = " Hi there! I'm Trang. I'm learning Natural Language Processing. Let's see how it works. "
print("Original Text: \n", text)
print("-"*60)
#NLTK Sentence Segmentation
sentences = sent_tokenize(text) #split text into sentences
print("NLTK Sentence Segmentation:")
for i, sent in enumerate(sentences, 1):
    print(f"Sentence {i}: {sent}")
print("-"*60)

#NLTK Word Tokenization
nltk_words = word_tokenize(text) #split text into words
print("NLTK Word Tokenization:")
print(nltk_words)
print("-"*60)
#nltk normalization
#Lowercasing all words
lowercased_words = [word.lower() for word in nltk_words]
print("Lowercased Words:")
print(lowercased_words)
print("-"*60)

#Removing punctuation
punctuation_removed_words = [word for word in lowercased_words if word not in string.punctuation]
print("Punctuation Removed Words:")
print(punctuation_removed_words)
print("-"*60)

#Reconstruct normalized text
normalized_text = ' '.join(punctuation_removed_words)
print("normalized text:\n", normalized_text)
print("-"*60)

#Stanza pos tagging
doc = nlp(normalized_text) #run stanza pipeline on normalized text
print("Stanza POS Tagging:")
for sentence in doc.sentences: #iterate through sentences
    for word in sentence.words: #iterate through words in each sentence
        print(f"Word: {word.text}\tPOS: {word.upos}")
print("-"*60)
#stanza depedency parsing
print("Stanza Dependency Parsing:")
for sentence in doc.sentences: 
    for word in sentence.words:
                print(f"Word: {word.text}\tHead: {word.head}\tDepRel: {word.deprel}")
print("-"*60)

#Stanza Constituency Parsing
# Note: Stanza does not have built-in constituency parsing; this is a placeholder
print("STANZA CONSTITUENCY PARSING:(Not available in Stanza)")
print("Stanza does not support constituency parsing directly.")
print("-"*60)
