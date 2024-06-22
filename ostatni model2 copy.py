import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer; 
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


tokenizer = get_tokenizer("basic_english")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



def embedd_list(lst):
    embedded_tens = np.zeros((199, 100))
    for i in range(199-len(lst),199):
        try:
         embedded_tens[i] = glove[lst[i-199+len(lst)]]
        except:
            pass       
    return embedded_tens



def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train_model(x_train, y_train, x_test, y_test, model, num_epochs, optimizer, criterion, batch_size):
    train_losses = []
    train_accs = []
    test_accs = []
    costs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch.float())
            loss = criterion(outputs, y_batch.squeeze_())
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * x_batch.size(0)
            epoch_train_acc += categorical_accuracy(outputs, y_batch.squeeze_()).item() * x_batch.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_acc /= len(train_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            outputs = model(x_test.float())
            test_acc = categorical_accuracy(outputs, y_test.squeeze_()).item()
            test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')
    
    return train_losses, train_accs, test_accs




def tokens(data):
    tokenized_data = []
    for sentence in data:
        sentence = tokenizer(sentence)
        tokenized_data.append(sentence)
    return tokenized_data


class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, hidden_dim=128, num_fc_layers=1):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.total_filters = len(filter_sizes) * n_filters        
        fc_layers = []
        input_dim = self.total_filters
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(input_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        fc_layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc_layers(cat)
    
    def predict(self, input):
        self.eval()
        with torch.no_grad():
            output = self.forward(input)
            return output




def evaluate(x,y,model,batch,iterator):
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        print(iterator)
        input()
        while iterator:
           loader = DataLoader(TensorDataset(x,y), batch_size= batch)
           for x_batch, y_batch in loader:
                predictions = model(x_batch.float())
            
                acc = categorical_accuracy(predictions, y_batch.squeeze_())
           epoch_acc += acc.item()
           iterator -= 1
           print("Evaluate Iteration:" + str(iterator))
        
    return epoch_acc / iterator



glove = dict()
embedding_dim = 100

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'glove6B100d.txt')

with open(file_path, encoding='utf-8') as fp:
    print(fp)
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        glove [word] = vector_dimensions


is_cuda = torch.cuda.is_available()

print(torch.version.cuda)

if is_cuda:
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")
    

file_path = os.path.join(current_directory, 'tcc_ceds_music.csv')


df = pd.read_csv(file_path)
df.head()


df = df[df.len > 4]

# Create a mapping from genre names to numbers
unique_genres = pd.unique(df['genre'])
genre_to_number = {genre: idx for idx, genre in enumerate(unique_genres)}
number_to_genre = {idx: genre for genre, idx in genre_to_number.items()}

# Replace genres with numbers
df['genre_num'] = df['genre'].map(genre_to_number)

# Prepare lyrics and topic dataframes
lyrics = df[['lyrics']]
lyrics['a'] = range(len(lyrics))
lyrics = lyrics.set_index('a')

topic = df[['genre_num']]

# Under-sample the data
rus = RandomUnderSampler()
lyrics_resampled, topic_resampled = rus.fit_resample(lyrics, topic)

# Print the count of each genre
print(topic_resampled['genre_num'].value_counts())

# Find the maximum length of lyrics
print(len(max(lyrics_resampled['lyrics'], key=len)))

# Print the mapping of numbers to genres
print(number_to_genre)



##################################################################################################
lyrics_resampled = tokens(lyrics_resampled['lyrics'])



X_train, X_test, y_train, y_test = train_test_split(lyrics_resampled, topic_resampled, test_size=0.2, random_state=42)

map_object_train = map(embedd_list, X_train)
X_train_embedded = list(map_object_train)
X_train_embedded = np.stack(X_train_embedded)
y_train = torch.from_numpy(y_train.values).to(device)
map_object_test = map(embedd_list, X_test)
X_test_embedded = list(map_object_test)
X_test_embedded = np.stack(X_test_embedded)
y_test = torch.from_numpy(y_test.values).to(device)

X_test_embedded = torch.from_numpy(X_test_embedded).to(device)
X_train_embedded = torch.from_numpy(X_train_embedded).to(device)

X_train_embedded = X_train_embedded.view(len(X_train_embedded), 1, 199, 100).to(device)
X_test_embedded = X_test_embedded.view(len(X_test_embedded), 1, 199, 100).to(device)

# model parameters
EMBEDDING_DIM = 100
N_FILTERS = 400
FILTER_SIZES = [2, 3, 4]
OUTPUT_DIM = 8
DROPOUT = 0.8
HIDDEN_DIM = 512
NUM_FC_LAYERS = 15

# init for program
model = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, HIDDEN_DIM, NUM_FC_LAYERS)
model.to(device)

# optimisation / criterion
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

# Train params
num_epochs = 300
batch_size = 30

#train
train_losses, train_accs, test_accs = train_model(X_train_embedded, y_train, X_test_embedded, y_test, model, num_epochs, optimizer, criterion, batch_size)

print(f"Length of train_losses: {len(train_losses)}")
print(f"Length of train_accs: {len(train_accs)}")
print(f"Length of test_accs: {len(test_accs)}")


# loss
torch.save(model, "C:\\Users\\szcza\\Desktop\\biai\\model")
plt.plot(range(len(train_losses)),train_losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# Plotting accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accs, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accs, label='Test Accuracy')
plt.title('Training and Test Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# confusion matrix
model.eval()
with torch.no_grad():
    X_test_embedded = X_test_embedded.float().to(device)
    Y_pred = model.predict(X_test_embedded)
    Y_pred_argmax = Y_pred.argmax(dim=1).cpu().numpy()

# Display confusion matrix
matrix = confusion_matrix(y_test.cpu().numpy(), Y_pred_argmax)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=number_to_genre.values())
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax)
plt.show()

print(classification_report(y_test.cpu().numpy(), Y_pred_argmax, target_names=number_to_genre.values()))