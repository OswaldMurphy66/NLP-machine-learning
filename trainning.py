import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import os

df_trainning = pd.read_excel('Dataset for NLP training and testing.xlsx', header=1, sheet_name=['201_labeled','203_labeled','204_labeled','205_labeled','209_labeled'])

df_testing = pd.read_excel('Dataset for NLP training and testing.xlsx', header=1, sheet_name=['202_predicted','206_predicted','207_predicted','208_predicted','210_predicted','211_predicted'])

#sheets = {sheet_name: df.parse(sheet_name) for sheet_name in df.sheet_names}

train_sentence = []
train_label = []
 
# iterate through all the sheets in Excel file
for sheet_name in df_trainning:
#    interate through all the rows in specific sheet
    for index, row in df_trainning[sheet_name].iterrows():
        train_sentence.append(row.values[0])
        a=row.values[6]
        # get the the first letter of the label
        # if a is Nan, then append 0
        if pd.isnull(a):
            o=int(row.values[9][0])
            train_label.append(int(row.values[9][0])) 
        else:
            o=int(row.values[6][0])
            train_label.append(int(row.values[6][0]))     
        
# in train_label, replace 3 with 2
for i in range(len(train_label)):
    if train_label[i]==3:
        train_label[i]=2

# in train_label, replace 2 with 0
for i in range(len(train_label)):
    if train_label[i]==2:
        train_label[i]=0
 
# extracting all sentences from testing data
test_sentence = []
for sheet_name in df_testing:
    for index, row in df_testing[sheet_name].iterrows():
        test_sentence.append(row.values[0])



print(test_sentence)

# load train_sentence and train_label to pytorch for trainning


# Assuming train_sentence and train_label are your data

# Step 1: Data Preparation
class SentenceDataset(Dataset):
    def __init__(self, sentences, labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

# 假设 train_sentence 和 train_label 是您的完整数据集
# 将数据集切分为训练集和验证集（30%作为验证集）
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentence, train_label, test_size=0.3, random_state=42)

# Convert your data into a Dataset
dataset = SentenceDataset(train_sentences, train_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Model Definition
class SentenceClassifier(nn.Module):
    def __init__(self, num_labels):
        super(SentenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialize the model
model = SentenceClassifier(num_labels=len(set(train_label)))
# if the model is already trained, load the weights
# if os.path.exists('./model_weights.pth'):
#     model.load_state_dict(torch.load('./model_weights.pth'))
# Count the number of unique classes
num_classes = len(set(train_label))
# Make sure the model's final layer matches this number
model = SentenceClassifier(num_labels=num_classes)

# Step 3: Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.train()  # set the model to training mode
    for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if i % 10 == 0:  # print every 10 mini-batches
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0

    epoch_accuracy = correct_predictions / total_predictions
    print(f'Epoch: {epoch + 1}, Training Accuracy: {epoch_accuracy}')


print('Finished Training')

# Save the model weights
torch.save(model.state_dict(), './model_weights.pth')


#evaluating the model

# Step 4: Evaluation
# Convert your validation data into a Dataset
val_dataset = SentenceDataset(val_sentences, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(model, dataloader):
    model.eval()  # set the model to evaluation mode
    predictions, true_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    # Calculate accuracy and other statistics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    return accuracy, precision, recall, f1

# Prepare the validation dataloader
val_dataset = SentenceDataset(val_sentences, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate(model, val_dataloader)
print(f"Validation Accuracy: {accuracy}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


