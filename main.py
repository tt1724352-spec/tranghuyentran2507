import nltk
import re
import string  
# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#input text
text = "My name is Trang. I'm 20 years old, I'm a third-year students at HUFLIT University and I learn English Teaching major."
print("Original Text:")
print(text)

# 1. Tokenization
token = word_tokenize(text)
print("TOKENS:\n", token)
print("-" * 60)

# 2. Lowercasing
lowercased_tokens = [word.lower() for word in token]
print("Lowercased Tokens:\n", lowercased_tokens)
print("-" * 60)

# 3. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in lowercased_tokens if word not in stop_words]
print("Tokens after Stopword Removal:\n", filtered_tokens)
print("-" * 60)

# 4. Punctuation Removal
punctuation_removed_tokens = [word for word in filtered_tokens if word not in string.punctuation]
print("Tokens after Punctuation Removal:\n", punctuation_removed_tokens)
print("-" * 60)

# 5. Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in punctuation_removed_tokens]
print("Stemmed Tokens:\n", stemmed_tokens)
print("-" * 60)

# 6. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmqtized_tokens = [lemmatizer.lemmatize(word) for word in punctuation_removed_tokens]
print("Lemmatized Tokens:\n", lemmatized_tokens)
print("-" * 60)

# Text Normalization Function
def nornalize_text(input_text):
    # Tokenization
    tokens = word_tokenize(input_text)

    # Lowercasing
    lowercased_tokens = [word.lower() for word in tokens]

    # Stopword Removal
    filtered_tokens = [word for word in lowercased if word not in stop_words]

    # Punctuation Removal
    punctuation_removed = [word for word in filtered if word not in string.punctuation]

    # Stemming
    stemmed = [stemmer .stem(word) for word in punctuation_removed]

    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in punctuation_removed]

return {
     "tokens": tokens,
     "lowercased": lowercased,
     "filtered": filtered,
     "punctuation_removed": punctuation_removed,
     "stemmed": stemmed,
     "lemmatized": lemmatized
}
print("Text Normalization Function Output:")
print(normalize_text(text))
print("-" * 60)
print("End of Text Normalization Process")