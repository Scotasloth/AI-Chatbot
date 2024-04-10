import torch
import torch.nn as nn
import torch.optim as optim

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
    
input_size = 100
hidden_size = 256
output_size = 100

model = Childebot(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)