import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download necessary NLTK data files if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Function to preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens if token.isalnum()]
    return tokens

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

def load_model(model_path):
    checkpoint = torch.load(model_path)
    vocab_size = len(checkpoint['word_to_idx'])
    embedding_dim = 100
    hidden_dim = 128
    output_dim = vocab_size
    
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['word_to_idx'], checkpoint['idx_to_word']

def predict_next_word(model, word_to_idx, idx_to_word, input_sentence, sequence_length=5):
    input_tokens = preprocess_text(input_sentence)
    print(f"Preprocessed tokens: {input_tokens}")  # Debugging: print preprocessed tokens
    
    if not input_tokens:
        return "No valid input tokens found."
    
    # Add '<UNK>' token handling
    unk_token = '<UNK>'
    if unk_token not in word_to_idx:
        word_to_idx[unk_token] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = unk_token
    
    input_sequence = [word_to_idx.get(token, word_to_idx[unk_token]) for token in input_tokens[-sequence_length:]]
    
    if len(input_sequence) == 0:
        return "Input sequence is empty after preprocessing."
    
    input_sequence = torch.tensor([input_sequence], dtype=torch.long)
    
    with torch.no_grad():
        output = model(input_sequence)
        print(f"Model output: {output}")  # Debugging: print model output
        
        predicted_index = torch.argmax(output, dim=1).item()
        print(f"Predicted index: {predicted_index}")  # Debugging: print predicted index
        
        predicted_word = idx_to_word.get(predicted_index, unk_token)
        return predicted_word

