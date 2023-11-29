import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def describe_value(value):
    """Describe the value based on defined ranges."""
    if value > 1:
        return "S"  # Strong growth
    elif value > 0:
        return "M"  # Moderate growth
    elif value > -1:
        return "D"  # Slight decline
    else:
        return "H"  # Sharp decline

def encode_row(row):
    """Encode a row of financial data into a 'word' in Candlestickish."""
    # Concatenating the first letter of each description
    return ''.join([describe_value(val) for val in row])

def encode_data(file_path):
    """Encode the entire dataset into Candlestickish language."""
    # Load the dataset
    data = pd.read_csv(file_path)

    # Apply encoding to each row
    encoded_data = data.apply(encode_row, axis=1)

    return encoded_data
# Step 1: Tokenization
#def create_token_mapping(encoded_data):
#    unique_words = set(encoded_data)
#    token_mapping = {word: i for i, word in enumerate(unique_words)}
#    return token_mapping
def create_token_mapping(encoded_data):
    unique_words = set(encoded_data[0])  # Pass the column of words to the set function
    token_mapping = {word: i for i, word in enumerate(unique_words)}
    return token_mapping

# Step 2: Sequence Formation
def create_sequences(encoded_data, sequence_length):
    sequences = []
    for i in range(len(encoded_data) - sequence_length):
        sequence = encoded_data[i:i + sequence_length]
        sequences.append(sequence)
    return sequences
file_path = 'memory.csv'  # Replace with the path to your dataset
#encoded_data = encode_data(file_path)
encoded_data = pd.read_csv('word_memory.csv', header=None)
print(encoded_data.head())  # Print the first few encoded words 
# Load your encoded data
# encoded_data = ... # Load your encoded data here

# Create token mapping
token_mapping = create_token_mapping(encoded_data)
print(token_mapping)

# Convert encoded words to tokens
#tokenized_data = [token_mapping[word] for word in encoded_data]
tokenized_data = [token_mapping[word] for word in encoded_data[0]]

# Create sequences
sequence_length = 100  # Example sequence length
sequences = create_sequences(tokenized_data, sequence_length)

# Step 3: Splitting Data
train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
validation_sequences, test_sequences = train_test_split(test_sequences, test_size=0.5, random_state=42)

class CandlestickDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]  # All tokens except the last
        target_seq = sequence[1:]  # All tokens except the first
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# Training DataLoader
train_dataset = CandlestickDataset(train_sequences)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
val_sequences, test_sequences = train_test_split(test_sequences, test_size=0.5, random_state=42)
# Validation DataLoader
val_dataset = CandlestickDataset(val_sequences)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Test DataLoader (if needed)
test_dataset = CandlestickDataset(test_sequences)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def plot_predictions(model, data_loader, token_mapping, device, epoch):
    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            predicted_tokens = torch.argmax(output, dim=2)
            actual.extend(target_seq.view(-1).cpu().numpy())
            predicted.extend(predicted_tokens.view(-1).cpu().numpy())

    plt.figure(figsize=(10, 4))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(f'Epoch {epoch + 1}')
    plt.xlabel('Sequence')
    plt.ylabel('Token')
    plt.legend()
    plt.show()

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size  # Set the embed_size as an instance attribute
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional._encoding = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embed_size, vocab_size)

        # Attempt to load a previously saved model
        try:
            self.load_model('model.pth', device)
        except FileNotFoundError:
            print("No model found")

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embed_size)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)
        return output

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
                             

# Model Hyperparameters
vocab_size = len(token_mapping)  # Number of unique tokens in your encoded data
embed_size = 512  # Embedding size
num_heads = 8  # Number of heads in multi-head attention
hidden_dim = 2048  # Dimension of the feedforward network
num_layers = 6  # Number of encoder layers
dropout = 0.1  # Dropout rate
model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout)
# Initialize the model
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for input_seq, target_seq in data_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        # Forward pass
        output = model(input_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(data_loader)
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq)
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            total_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            print(predicted.item())

    return total_loss / len(data_loader)
num_epochs = 100
# Example usage in the training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plotting predicted vs actual values
    #plot_predictions(model, val_loader, token_mapping, device, epoch)

    model.save_model(f'model.pth')