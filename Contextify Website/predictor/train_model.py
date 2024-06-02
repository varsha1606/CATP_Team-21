import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from predictor.preprocessing import preprocess_text, load_and_preprocess  # absolute import
from predictor.lstm_model import LSTMModel

# Define the custom dataset class
class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# Training function
def train_and_save_model(file_paths, model_path):
    tokens = load_and_preprocess(file_paths)
    vocab = list(set(tokens))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    sequences = []
    targets = []
    sequence_length = 5
    
    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length]
        target = tokens[i + sequence_length]
        sequences.append([word_to_idx[word] for word in seq])
        targets.append(word_to_idx[target])

    dataset = TextDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = vocab_size
    
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
    # Save the model and dictionaries
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word
    }, model_path)

if __name__ == "__main__":
    file_paths = ["C:/Users/varsh/Downloads/Sherlock Holmes.txt"]
    model_path = "lstm_model.pth"
    train_and_save_model(file_paths, model_path)
