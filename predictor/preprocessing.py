import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data files if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Load and preprocess text files
def load_and_preprocess(file_paths):
    tokens = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        tokens.extend(preprocess_text(text))
    return tokens

if __name__ == "__main__":
    # Example usage
    file_paths = ["C:/Users/varsh/Downloads/Sherlock Holmes.txt"]
    tokens = load_and_preprocess(file_paths)
    print(tokens[:100])  # Print the first 100 tokens as a sample
