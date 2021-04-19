import numpy as np
from datetime import datetime
from time import time
import torch
from torch import nn, optim, Tensor, transpose
import matplotlib.pyplot as plt

from data_preprocessing import load_stock_price_data, load_tweet_data, combine_price_and_sentiment, gen_timeseries_samples, normalize_features
from sentiment_analysis import pretrained_sentiment
from utils import positional_encoding, recover_true_values, elapsed_time


is_rnn = False

if is_rnn:
    epochs = 100
    batch_size = 16
    learning_rate = 5e-2
else:
    epochs = 100
    batch_size = 16
    learning_rate = 5e-4
# Data parameters
step = 5
lag = 4
d_model = 7
# Date parameters
start_dt = datetime(2014, 1, 1)
end_train = datetime(2015, 7, 31)
end_val = datetime(2015, 9, 30)
end_dt = datetime(2016, 1, 1)
# Model parameters
if is_rnn:
    model = nn.RNN(input_size=d_model, hidden_size=d_model, num_layers=1).to('cuda')
else:
    model = nn.Transformer(d_model, nhead=d_model, num_encoder_layers=12, num_decoder_layers=12).to('cuda')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss(reduction='mean')
positional_encoder = torch.tensor(positional_encoding(lag, d_model)).to('cuda')


print('Beginning Preprocessing:')
start_preprocessing = time()

print('\t(1/6) Loading stock price data...')
price_data_train, price_data_val, price_data_test = load_stock_price_data('/home/mrkeaton/Documents/Datasets/stocknet-dataset/price/preprocessed', start_dt, end_train, end_val, end_dt)
print('\t(2/6) Loading tweet data...')
tweet_data_train, tweet_data_val, tweet_data_test = load_tweet_data('/home/mrkeaton/Documents/Datasets/stocknet-dataset/tweet/preprocessed', end_train, end_val)

print('\t(3/6) Generating sentiments from tweet data...')
sentiments_train = pretrained_sentiment(tweet_data_train)
sentiments_val = pretrained_sentiment(tweet_data_val)
sentiments_test = pretrained_sentiment(tweet_data_test)

print('\t(4/6) Combining price data and sentiments...')
d_train = combine_price_and_sentiment(price_data_train, sentiments_train, start_dt, end_train)
d_val = combine_price_and_sentiment(price_data_val, sentiments_val, end_train, end_val)
d_test = combine_price_and_sentiment(price_data_test, sentiments_test, end_val, end_dt)

print('\t(5/6) Creating time-lagged data samples...')
data_train = gen_timeseries_samples(d_train, step, lag)
data_val = gen_timeseries_samples(d_val, step, lag)
data_test = gen_timeseries_samples(d_test, step, lag)

print('\t(6/6) Normalizing data...')
norm_data_train, min_dtrain, max_dtrain = normalize_features(data_train)
norm_data_val, min_dval, max_dval = normalize_features(data_val)
norm_data_test, min_dtest, max_dtest = normalize_features(data_test)

print('Completed in {}\n'.format(elapsed_time(time() - start_preprocessing)))

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
    print('Epoch {}'.format(e))
    print('Training:', end=' ')
    rng.shuffle(norm_data_train)
    for batch in range(0, len(norm_data_train), batch_size):
        model.zero_grad()
        data = norm_data_train[batch:batch+batch_size]
        labels = Tensor(data[np.newaxis, :, 4]).to('cuda')
        data = transpose(Tensor(data[:, :4]), 0, 1).to('cuda')
        if not is_rnn:
            data += positional_encoder[:, None, :]
        output = model(data, labels)
        if is_rnn:
            output = output[1]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        recovered_labels = recover_true_values(labels, min_dtrain, max_dtrain)
        recovered_output = recover_true_values(output, min_dtrain, max_dtrain)
        for i in range(labels.shape[1]):
            if (recovered_labels[0][i][0] >= 0 and recovered_output[0][i][0] >= 0) or (recovered_labels[0][i][0] <= 0 and recovered_output[0][i][0] <= 0):
                train_acc += 1

    train_acc /= len(norm_data_train)
    train_accs.append(train_acc)
    train_loss /= len(norm_data_train)
    train_losses.append(train_loss)
    print('{} Loss: {}; Accuracy: {}'.format(elapsed_time(time() - start_train), train_loss, train_acc))

    start_val = time()
    model.eval()
    val_acc = 0
    val_loss = 0

    print('Testing:', end=' ')
    for batch in range(0, len(norm_data_val), batch_size):
        data = norm_data_val[batch: batch+batch_size]
        labels = Tensor(data[np.newaxis, :, 4]).to('cuda')
        data = transpose(Tensor(data[:, :4]), 0, 1).to('cuda')
        if not is_rnn:
            data += positional_encoder[:, None, :]
        output = model(data, labels)
        if is_rnn:
            output = output[1]
        loss = criterion(output, labels)
        val_loss += loss.item()
        recovered_labels = recover_true_values(labels, min_dval, max_dval)
        recovered_output = recover_true_values(output, min_dval, max_dval)
        for i in range(labels.shape[1]):
            if (recovered_labels[0][i][0] >= 0 and recovered_output[0][i][0] >= 0) or (
                        recovered_labels[0][i][0] <= 0 and recovered_output[0][i][0] <= 0):
                val_acc += 1

    val_acc /= len(norm_data_val)
    val_accs.append(val_acc)
    val_loss /= len(norm_data_val)
    val_losses.append(val_loss)
    print('{} Loss: {}; Accuracy: {}'.format(elapsed_time(time() - start_val), val_loss, val_acc))

print('Testing:')
start_test = time()
test_accs = []
test_acc = 0
# test_losses = []
# test_loss = 0
for batch in range(0, len(norm_data_test), batch_size):
    data = norm_data_test[batch: batch+batch_size]
    labels = Tensor(data[np.newaxis, :, 4]).to('cuda')
    data = transpose(Tensor(data[:, :4]), 0, 1).to('cuda')
    if not is_rnn:
        data += positional_encoder[:, None, :]
    output = model(data, labels)
    if is_rnn:
        output = output[1]
    # loss = criterion(output, labels)
    # test_loss += loss.item()
    recovered_labels = recover_true_values(labels, min_dtest, max_dtest)
    recovered_output = recover_true_values(output, min_dtest, max_dtest)
    for i in range(labels.shape[1]):
        if (recovered_labels[0][i][0] >= 0 and recovered_output[0][i][0] >= 0) or (
                    recovered_labels[0][i][0] <= 0 and recovered_output[0][i][0] <= 0):
            test_acc += 1
test_acc /= len(norm_data_test)
test_accs.append(test_acc)
# val_loss /= len(norm_data_val)
# val_losses.append(val_loss)
print('{} Accuracy: {}'.format(elapsed_time(time() - start_test), test_acc))

plt.figure()
plt.plot(range(1, epochs+1), train_losses)
plt.plot(range(1, epochs+1), val_losses)
plt.legend(('Training Losses', 'Validation Losses'))
if is_rnn:
    plt.title('Training and Evaluation Losses for RNN Model')
else:
    plt.title('Training and Evaluation Losses for Baseline Transformer Network')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(range(1, epochs+1), train_accs)
plt.plot(range(1, epochs+1), val_accs)
plt.legend(('Training Accuracy', 'Validation Accuracy'))
if is_rnn:
    plt.title('Training and Evaluation Accuracies for RNN Model')
else:
    plt.title('Training and Evaluation Accuracies for Baseline Transformer Network')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

print('test')
