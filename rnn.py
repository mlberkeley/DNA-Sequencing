import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn
import sklearn.model_selection
import random

# read the excel sheet 
df = pd.read_excel('./BioCode for Machine Learning Updated.xlsx')

# Read in the labels
cls = df['Classification']

# Read the DNA sequences, which are strings comprised of the letters ATCG
seq = df['Aligned Sequence']

Order_group = df['Order_Group']

# O_g = np.array([o_g.split('_') for o_g in Order_group])

species = df['Species']

# Convert DNA data to numpy array, and convert NaNs to Nones
seq = np.array(seq.fillna('None'))

# Create a binary filter to eliminate invalid DNA sequences
valid_idx = np.array([i for i in range(len(seq)) if seq[i] != 'None'])

# Apply the filter
valid_seq = seq[valid_idx]
cls_valid = cls[valid_idx]
cls_valid = np.array(cls_valid)

# Seperate string into individual characters
seq_arrays = valid_seq #[np.array([i for i in s]) for s in valid_seq]

valid_labels = ['Introduced', 'Invasive', 'Indigenous']
labeled_cls = [label in valid_labels for label in cls_valid]
#labeled_cls = (valid_labels[labeled_cls] == 'Indigenous').astype(int)

# Create a filter telling us which points are valid to use for supervised training
labeled_cls = np.array(labeled_cls)

#print(len(labeled_cls))
cls_valid[labeled_cls]

# apply the filter over our features and labels
supervised_X = seq_arrays[labeled_cls]
supervised_y = cls_valid[labeled_cls]

supervised_y = (supervised_y == 'Indigenous').astype(int)

supervised_X, test_X, supervised_y, test_y = sklearn.model_selection.train_test_split(supervised_X, supervised_y)

n_categories = 2 # len(all_categories)
n_letters = 5

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(base):
    if base == 'A':
        return 0
    elif base == 'T':
        return 1
    elif base == 'C':
        return 2
    elif base == 'G':
        return 3
    else:
        return 4


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def linesToTensor(lines):
    tensor = torch.zeros(len(lines[0]), len(lines), n_letters)
    for i, line in enumerate(lines):
        for li, letter in enumerate(line):
            tensor[li][i][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(5, hidden_size, 2)
        
        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, 100)
        self.i2o2 = nn.Linear(100, output_size)
        self.softmax = nn.LogSoftmax()

        self.h0 = Variable(torch.zeros(2, 16, self.hidden_size).cuda())
        self.c0 = Variable(torch.zeros(2, 16, self.hidden_size).cuda())

        self.test_h0 = Variable(torch.zeros(2, 421, self.hidden_size).cuda())
        self.test_c0 = Variable(torch.zeros(2, 421, self.hidden_size).cuda())
    
    def forward(self, input, testing=False):
        # combined = torch.cat((input, hidden), 1)
        # h0 = Variable(torch.zeros(2, 16, self.hidden_size))
        # c0 = Variable(torch.zeros(2, 16, self.hidden_size))

        print(type(input.data), input.size())

        if testing:
            output, hn = self.lstm(input, (self.test_h0, self.test_c0))
        else:
            output, hn = self.lstm(input, (self.h0, self.c0))

        output = output[-1,:,:]

        # hidden = self.i2h(combined)
        output = self.i2o(output)
        output = self.i2o2(output)
        output = self.softmax(output)
        return output


n_hidden = 1024
rnn = RNN(n_letters, n_hidden, n_categories)

rnn.cuda()

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return bool(category_i), category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(num, X, y):
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
#     line_tensor = Variable(lineToTensor(line))
    rng = np.random.choice(len(X), size=(num), replace=False)
    category = y[rng]
    line = X[rng]
    category_tensor = Variable(torch.LongTensor(category)).cuda()
    line_tensor = Variable(linesToTensor(line)).cuda()
    # print(category_tensor.size(), line_tensor.size())
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    # hidden = rnn.initHidden()

    rnn.zero_grad()

    # print(line_tensor.size())

    output = rnn(line_tensor)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

def test(category_tensor, line_tensor):
    rnn.zero_grad()

    output = rnn(line_tensor, testing=True)

    total = 0
    correct = 0

    for gt, pred in zip(category_tensor, output):
        total += 1
        correct += gt == categoryFromOutput(pred)[1]

    acc = correct / float(total)

    return acc

n_iters = 100
print_every = 1
plot_every = 5


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(16, supervised_X, supervised_y)
    # print(category_tensor.size(), line_tensor.size())
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # # Print iter number, loss, name and guess
    # if iter % print_every == 0:
    #     guess, guess_i = categoryFromOutput(output)
    #     correct = '✓' if guess == category else '✗ (%s)' % category
    #     print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        print('testing', loss, iter)

    if iter % print_every == 0:
        category, line, category_tensor, line_tensor = randomTrainingExample(len(test_X), test_X, test_y)
        acc = test(category_tensor, line_tensor)
        print('testing:', acc)


plt.figure()
plt.plot(all_losses)


