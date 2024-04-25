import nltk
nltk.download('punkt')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
stop_words = set(stopwords.words('english'))


embeddings = torch.load('svd-word-vectors.pt')


def preprocess_text(text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        return tokens


def collate_fn(batch):
    inputs, labels = zip(*batch)  
    inputs = pad_sequence(inputs, batch_first=True)  
    labels = torch.tensor(labels, dtype=torch.long)  
    return inputs, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_embeddings):
        self.texts = texts
        self.labels = labels
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        embedding = torch.tensor([self.word_embeddings[word] for word in text if word in self.word_embeddings], dtype=torch.float)
        return embedding, label



train_df = pd.read_csv('train.csv')
train_df['Processed_Description'] = train_df['Description'].apply(preprocess_text)
label_encoder = LabelEncoder()
train_df['Class Index'] = label_encoder.fit_transform(train_df['Class Index'])
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_data['Processed_Description'].tolist(), train_data['Class Index'].tolist(), embeddings)
val_dataset = TextDataset(val_data['Processed_Description'].tolist(), val_data['Class Index'].tolist(), embeddings)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)







class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.fc(ht[-1])

model = LSTMClassifier(100, 256, len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



def validate_model(model, val_loader, criterion, device):
    model.eval()  
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

epochs = 3
for epoch in range(epochs):
    model.train() 
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

def evaluate_model(model, data_loader, device):
    model.eval() 
    predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.view(-1).cpu().numpy())
            true_labels.extend(labels.view(-1).cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)

    return accuracy, precision, recall, f1, conf_matrix  


def print_metrics(metrics):
    accuracy, precision, recall, f1, conf_matrix = metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)





label_encoder = LabelEncoder()
test_df = pd.read_csv('test.csv')
test_df['Processed_Description'] = test_df['Description'].apply(preprocess_text)
test_df['Class Index'] = label_encoder.fit_transform(test_df['Class Index'])
test_dataset = TextDataset(test_df['Processed_Description'].tolist(), test_df['Class Index'].tolist(), embeddings)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


model_save_path = './svd-classification-model.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')



model = LSTMClassifier(100, 256, len(label_encoder.classes_))
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)
print('Model loaded successfully.')


train_metrics = evaluate_model(model, train_loader, device)
test_metrics = evaluate_model(model, test_loader, device)


print_metrics(train_metrics)
print_metrics(test_metrics)