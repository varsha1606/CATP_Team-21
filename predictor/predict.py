import torch
from predictor.lstm_model import LSTMModel
from predictor.preprocessing import preprocess_text

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
