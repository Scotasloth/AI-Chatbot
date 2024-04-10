import torch
import torch.nn as nn
import torch.optim as optim
import sys

class Childebot(nn.Module):

    def  __int__(self, input_size, hidden_size, output_size):
        super(Childebot, self).__init__()
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)                
                           
    def forward(self, input):
        embedded = self.encoder(input)
        output, hidden = self.rnn(embedded)
        output - self.decoder(output)

        return output, hidden

def tokenize(text):
    return text.lower().split()

def get_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            input_text = line.strip().split('\t')
            target_text = line.strip().split('\t') 
            input_tokens = tokenize(input_text)
            target_tokens = tokenize(target_text)
            dataset.append((input_tokens, target_tokens))
    return dataset

dir = sys.path[0]
data_file = (dir + "\dataset.json")

dataset = get_dataset(data_file)

input_size = 100
hidden_size = 256
output_size = 100

model = Childebot(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for input, target in dataset:
    optimizer.zero_grad()
    output, _ = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    torch.save(model, 'Childebot.pth')