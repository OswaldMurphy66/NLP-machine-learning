
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# Define the fields for sentence and label
sentence_field = Field(sequential=True, tokenize='spacy', lower=True)
label_field = Field(sequential=False, use_vocab=False)

# Load the Excel file
data = TabularDataset(path='Dataset for NLP training and testing.xlsx', format='xlsx', skip_header=True,
                      fields=[('sentence', sentence_field), ('label', label_field)])

# Split the data into train and test sets
train_data, test_data = data.split(split_ratio=0.8)

# Build the vocabulary
sentence_field.build_vocab(train_data)

# Define the model
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output = self.fc(embedded)
        return output

# Initialize the model
input_dim = len(sentence_field.vocab)
hidden_dim = 128
output_dim = 2  # Assuming binary classification
model = Classifier(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Create iterators for batching the data
batch_size = 32
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, sort_key=lambda x: len(x.sentence), shuffle=True)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.sentence)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'model.pt')
