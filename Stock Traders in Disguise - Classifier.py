import os
import numpy as np
from datetime import datetime
from time import time
import torch
from torch import nn, optim, Tensor
import matplotlib.pyplot as plt
import pickle

from transformer_classifier import TransformerClassifier
from data_preprocessing import load_stock_price_data, load_tweet_data, combine_price_and_sentiment, gen_timeseries_samples, normalize_features
from sentiment_analysis import pretrained_sentiment
from utils import positional_encoding, recover_true_values, elapsed_time

version = 5
load_data = True


epochs = 20
batch_size = 16
learning_rate = 1e-6
# Data parameters
step = 4
lag = 7
d_model = 7
# Date parameters
start_dt = datetime(2014, 1, 1)
end_train = datetime(2015, 7, 31)
end_val = datetime(2015, 9, 30)
end_dt = datetime(2016, 1, 1)
# Model parameters
model = TransformerClassifier(d_model, d_model, lag, 2048, 12, 0.5).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()
positional_encoder = torch.tensor(positional_encoding(lag, d_model)).to('cuda')


def preprocess_data():
    print('Beginning Preprocessing:')
    start_preprocessing = time()

    print('\t(1/5) Loading stock price data...')
    price_data_train, price_data_val, price_data_test = load_stock_price_data(
        '/home/mrkeaton/Documents/Datasets/stocknet-dataset/price/preprocessed', start_dt, end_train, end_val, end_dt)

    print('\t(2/5) Loading tweet data...')
    tweet_data_train, tweet_data_val, tweet_data_test = load_tweet_data(
        '/home/mrkeaton/Documents/Datasets/stocknet-dataset/tweet/preprocessed', end_train, end_val)

    print('\t(3/5) Generating sentiments from tweet data...')
    sentiments_train = pretrained_sentiment(tweet_data_train)
    sentiments_val = pretrained_sentiment(tweet_data_val)
    sentiments_test = pretrained_sentiment(tweet_data_test)

    print('\t(4/5) Combining price data and sentiments...')
    d_train = combine_price_and_sentiment(price_data_train, sentiments_train, start_dt, end_train)
    d_val = combine_price_and_sentiment(price_data_val, sentiments_val, end_train, end_val)
    d_test = combine_price_and_sentiment(price_data_test, sentiments_test, end_val, end_dt)

    print('\t(5/5) Creating time-lagged data samples...')
    data_train = gen_timeseries_samples(d_train, step, lag)
    data_val = gen_timeseries_samples(d_val, step, lag)
    data_test = gen_timeseries_samples(d_test, step, lag)

    print('Completed in {}'.format(elapsed_time(time() - start_preprocessing)))

    print('Saving files as pickles...\n')
    if not os.path.exists('saved_data'):
        os.mkdir('saved_data')
    with open(os.path.join('saved_data', 'transformer_train_data.pkl'), 'wb') as train_d:
        pickle.dump(data_train, train_d)
    with open(os.path.join('saved_data', 'transformer_val_data.pkl'), 'wb') as val_d:
        pickle.dump(data_val, val_d)
    with open(os.path.join('saved_data', 'transformer_test_data.pkl'), 'wb') as test_d:
        pickle.dump(data_test, test_d)

    return data_train, data_val, data_test


if load_data:
    assert os.path.exists(os.path.join('saved_data', 'transformer_train_data.pkl'))\
           and os.path.exists(os.path.join('saved_data', 'transformer_val_data.pkl'))\
           and os.path.exists(os.path.join('saved_data', 'transformer_test_data.pkl')),\
           'train_data.pkl, val_data.pkl, or test_data.pkl do not exist.'
    with open(os.path.join('saved_data', 'transformer_train_data.pkl'), 'rb') as train_d:
        data_train = pickle.load(train_d)
    with open(os.path.join('saved_data', 'transformer_val_data.pkl'), 'rb') as val_d:
        data_val = pickle.load(val_d)
    with open(os.path.join('saved_data', 'transformer_test_data.pkl'), 'rb') as test_d:
        data_test = pickle.load(test_d)
else:
    data_train, data_val, data_test = preprocess_data()

print('Normalizing data...')
data_train, data_val, data_test, min_d, max_d = normalize_features(data_train, data_val, data_test)

rng = np.random.default_rng()
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for e in range(1, epochs + 1):
    start_train = time()
    model.train()
    train_acc = 0
    train_loss = 0
    print('Epoch {}/{} - Learning rate: {}'.format(e, epochs, optimizer.param_groups[0]['lr']))
    print('Training:', end=' ')
    rng.shuffle(data_train)
    for batch in range(0, len(data_train), batch_size):
        model.zero_grad()
        data = data_train[batch:batch+batch_size]
        labels = Tensor(data[:, lag]).to('cuda')
        rl = recover_true_values(labels, min_d, max_d)[0]
        target = [int(rl[i][0] > 0) for i in range(len(rl))]
        target = Tensor(target).long().to('cuda')
        data = Tensor(data[:, :lag]).to('cuda')
        data += positional_encoder[None, :, :]
        output = model(data, None)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        o = output.max(dim=1)[1]
        for i in range(len(target)):
            if target[i] == o[i]:
                train_acc += 1

    scheduler.step()
    train_acc /= len(data_train)
    train_accs.append(train_acc)
    train_loss /= len(data_train)
    train_losses.append(train_loss)
    print('{} Loss: {}; Accuracy: {}'.format(elapsed_time(time() - start_train), train_loss, train_acc))

    start_val = time()
    model.eval()
    val_acc = 0
    val_loss = 0

    print('Evaluating:', end=' ')
    for batch in range(0, len(data_val), batch_size):
        data = data_val[batch: batch+batch_size]
        labels = Tensor(data[:, lag]).to('cuda')
        rl = recover_true_values(labels, min_d, max_d)[0]
        target = [int(rl[i][0] > 0) for i in range(len(rl))]
        target = Tensor(target).long().to('cuda')
        data = Tensor(data[:, :lag]).to('cuda')
        data += positional_encoder[None, :, :]
        output = model(data, None)
        loss = criterion(output, target)
        val_loss += loss.item()
        o = output.max(dim=1)[1]
        for i in range(len(target)):
            if target[i] == o[i]:
                val_acc += 1

    val_acc /= len(data_val)
    val_accs.append(val_acc)
    val_loss /= len(data_val)
    val_losses.append(val_loss)
    print('{} Loss: {}; Accuracy: {}'.format(elapsed_time(time() - start_val), val_loss, val_acc))
    scheduler.step()

print('\nTesting:')
start_test = time()
test_acc = 0
test_loss = 0
for batch in range(0, len(data_test), batch_size):

    data = data_test[batch: batch + batch_size]
    labels = Tensor(data[:, lag]).to('cuda')
    rl = recover_true_values(labels, min_d, max_d)[0]
    target = [int(rl[i][0] > 0) for i in range(len(rl))]
    target = Tensor(target).long().to('cuda')
    data = Tensor(data[:, :lag]).to('cuda')
    data += positional_encoder[None, :, :]
    output = model(data, None)
    loss = criterion(output, target)
    test_loss += loss.item()
    o = output.max(dim=1)[1]
    for i in range(len(target)):
        if target[i] == o[i]:
            test_acc += 1

if not os.path.exists('results'):
    os.mkdir('results')

test_acc /= len(data_test)
test_loss /= len(data_test)
print('{} Accuracy: {}'.format(elapsed_time(time() - start_test), test_acc))

plt.figure()
plt.plot(range(1, epochs+1), train_losses)
plt.plot(range(1, epochs+1), val_losses)
plt.legend(('Training Losses', 'Validation Losses'))

plt.title('Training and Evaluation Losses for Transformer Network')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(os.path.join('results', 'transformer_trainval_loss_{}.pdf'.format(version)))

plt.figure()
plt.plot(range(1, epochs+1), train_accs)
plt.plot(range(1, epochs+1), val_accs)
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.title('Training and Evaluation Accuracies for Transformer Network')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join('results', 'transformer_trainval_acc_{}.pdf'.format(version)))

with open(os.path.join('results', 'transformer_overall_results_{}.txt'.format(version)), 'w') as wt:
    wt.write('Test accuracy: {}\n'.format(test_acc))
    wt.write('Test loss: {}\n'.format(test_loss))
    wt.write('Hyperparameters: Epochs {}; Batch size: {}; Learning rate: {}; step: {}; lag: {}\n'
             .format(epochs, batch_size, learning_rate, step, lag))
    wt.write('Scheduler: {}\n'.format(scheduler))
    wt.write('Loss: {}\n\n'.format(criterion))
    wt.write('Model: {}'.format(model))
