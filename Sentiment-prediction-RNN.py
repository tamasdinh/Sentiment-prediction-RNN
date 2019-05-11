#%%
import os
from string import punctuation
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from datetime import datetime

#%%
with open(os.getcwd() + '/data/reviews.txt', 'r') as f:
    reviews = f.read()
with open(os.getcwd() + '/data/labels.txt', 'r') as f:
    labels = f.read()

#%%
print(reviews[:2000])
print('--')
print(labels[:20])

#%%
# Text preprocessing
reviews = reviews.lower()
reviews = ''.join([ch for ch in reviews if ch not in punctuation])
reviews = reviews.split('\n')[:-1]
reviews = [rev.split() for rev in reviews]
words = []
for review in reviews:
    words += review

#%%
word_counts = Counter(words)
word2int = {word: idx for idx, word in enumerate(word_counts)}
assert len(word2int) == len(set(words)), 'It seems that the encoding dictionary does not have the same number of vocab' \
                                    'words as the raw data'
reviews_enc = [[word2int[word] for word in rev] for rev in reviews]

#%%
lab_dict = {'positive': 1, 'negative': 0}
labels = labels.split('\n')[:-1]
labels_enc = np.array([lab_dict[lab] for lab in labels])
assert len(labels_enc) == len(reviews_enc), 'It seems that the number of labels does not match the number of reviews'

#%%
review_ls = Counter([len(x) for x in reviews_enc])
print('Zero-length revs: {}'.format(review_ls[0]))
print('Longest review: {}'.format(max(review_ls)))


#%%
def pad_features(reviews_enc, seq_length):
    padded_revs = []
    for review in reviews_enc:
        act_len = len(review)
        if act_len >= seq_length:
            review = review[:seq_length]
        else:
            num_0s = seq_length - act_len
            review = [0 for i in range(num_0s)] + review
        padded_revs.append(review)
    return np.array(padded_revs)


#%%
test = pad_features(reviews_enc[:10], 200)
for rev in test:
    assert len(rev) == 200, 'It seems that the length of the padded reviews does not equal the desired seq_length'

#%%
padded_revs = pad_features(reviews_enc, 200)

#%%
split_frac = 0.8
train_idx = range(int(split_frac * padded_revs.shape[0]))
valid_idx = range(train_idx[-1] + 1, train_idx[-1] + (padded_revs.shape[0] - train_idx[-1] + 1) // 2)
test_idx = range(valid_idx[-1] + 1, padded_revs.shape[0])
print('Train index:\t{}'.format(train_idx))
print('Validation index:\t{}'.format(valid_idx))
print('Test index:\t{}'.format(test_idx))

#%%
train_x = padded_revs[train_idx]
valid_x = padded_revs[valid_idx]
test_x = padded_revs[test_idx]
train_y = labels_enc[train_idx]
valid_y = labels_enc[valid_idx]
test_y = labels_enc[test_idx]

for dataset in [train_x, valid_x, test_x, train_y, valid_y, test_y]:
    print(dataset.shape)

#%%
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

#%%
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: {}'.format(sample_x.size()))
print('Sample input: \n', sample_x)
print()
print('Sample output size: {}'.format(sample_y.size()))
print('Sample output: \n', sample_y)

#%%
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('\nTRAINING ON GPU')
else:
    print('\nNO GPU AVAILABLE -- TRAINING ON CPU...')


#%%
class SentimentRNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        x = self.embed(x)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.linear(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


#%%
# Instantiating the network
vocab_size = len(word2int)
output_size = 2
embedding_dim = 300
hidden_dim = 512
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

#%%
lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

#%%
epochs = 1
counter = 0
print_every = 100
clip = 5

if train_on_gpu:
    net.cuda()

net.train()

# epoch loop
for e in range(epochs):

    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        h = tuple([each.data for each in h])

        net.zero_grad()

        output, h = net(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in h])
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

            net.train()
            print('Epoch: {}/{}...'.format(e + 1, epochs),
                  'Step: {}...'.format(counter),
                  'Training loss: {:.4f}...'.format(loss.item()),
                  'Validation loss: {:.4f}'.format(np.mean(val_losses)))

#%%
model_name = 'sentiment-prediction-rnn_{}_{}.net'.format(datetime.today().date(), int(datetime.today().timestamp()))

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict()}

with open(f'./{model_name}', 'wb') as f:
    torch.save(checkpoint, f)

#%%
