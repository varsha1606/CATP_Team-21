import torch
from predictor.lstm_model import LSTMModel

def load_model(model_path):
    try:
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded successfully")  # Debugging

        vocab_size = len(checkpoint['word_to_idx'])
        embedding_dim = 100
        hidden_dim = 128
        output_dim = vocab_size

        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded and set to evaluation mode")  # Debugging

        return model, checkpoint['word_to_idx'], checkpoint['idx_to_word']
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")  # Debugging
        return None, None, None
