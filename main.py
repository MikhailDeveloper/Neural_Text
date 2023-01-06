from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TRAINS_FOLDER_FILE_PATH = 'training_files/'
MODELS_FOLDER_FILE_PATH = 'models_files/'
TRAIN_TEXT_FILE_DEFAULT_NAME = 'poetic_data'

SEQ_LEN = 256
BATCH_SIZE = 16


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    # Nprint(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char


def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start: batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def module_name(epochs):
    return MODELS_FOLDER_FILE_PATH + train_name + '_epoch_' + str(epochs) + '.pth'

def learn(epochs):
    model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )

    n_epochs = epochs  # 50000
    loss_avg = []

    for epoch in range(n_epochs):
        model.train()
        train, target = get_batch(sequence)
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        hidden = model.init_hidden(BATCH_SIZE)

        output, hidden = model(train, hidden)
        loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_avg.append(loss.item())
        if len(loss_avg) >= 50:
            mean_loss = np.mean(loss_avg)
            print(f'Loss: {mean_loss} Epoch: {epoch}')
            scheduler.step(mean_loss)
            loss_avg = []
            model.eval()
            predicted_text = evaluate(model, char_to_idx, idx_to_char)
            # print(predicted_text)

    torch.save(model,  module_name(epochs))

quer = input('Учить нейросеть? (Y/N): ')

train_name = input('Название датасета (без расширения): ')
if train_name == '':
    train_name = TRAIN_TEXT_FILE_DEFAULT_NAME

with open(TRAINS_FOLDER_FILE_PATH + train_name + '.txt') as text_file:
    text_sample = text_file.readlines()
text_sample = ' '.join(text_sample)
sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)

if (quer == 'Y'):
    epochs = int(input('Введите колличество эпох обучения (>1000): '))
    learn(epochs)
elif (quer == 'N'):
    epochs = int(input('Введите эпоху входной модели: '))
    model = torch.load(module_name(epochs))
    model.eval()
    while True:
        count = int(input('Введите колличество символов текста: '))
        N = int(input('Введите колличество генераций текста: '))
        for i in range(N):
            print('-----------------------------')
            print(evaluate(
                model,
                char_to_idx,
                idx_to_char,
                temp=0.3,
                prediction_len=count,
                start_text=' '
               )
            )
            print('-----------------------------\n\n')
        qrep = input('Еще разок? (Y/N): ')
        if (qrep != 'Y'): break
else:
    print('Ошибка ввода.')
