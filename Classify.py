import pandas as pd 
import torch 
from transformers import BertTokenizer  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

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

df_testing = pd.read_excel('Dataset for NLP training and testing.xlsx', header=1, sheet_name=['202_predicted','206_predicted','207_predicted','208_predicted','210_predicted','211_predicted'])

# Assuming model and SentenceClassifier are already defined
# Initialize and load your trained model
model = SentenceClassifier(num_labels=2)
model.load_state_dict(torch.load('./model_weights.pth'))
model.eval()  # set the model to evaluation mode

# Prepare the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to predict a single sentence
def predict(sentence, model):
    with torch.no_grad():
        inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        
        return prediction.item()
    
 
# iterate through all the sheets in Excel file
for sheet_name in df_testing:
#    interate through all the rows in specific sheet
    for index, row in df_testing[sheet_name].iterrows():
        sentence=row.values[0]
        prediction = predict(sentence, model)
        print(f"Sentence: {sentence}, Prediction: {prediction}")
        # write the prediction to raw Prediction column
        #  （1: requirment, 2 :information）
        txt='1 - Requirement' if prediction==1 else '2 - Information'
        df_testing[sheet_name].loc[index,'Prediction']=txt

# save the result original excel file
with pd.ExcelWriter('result.xlsx') as writer:
    for sheet_name in df_testing:
        df_testing[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
        

        