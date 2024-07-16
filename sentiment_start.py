########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.functional import pad

import loader as ld

batch_size = 32
output_size = 2
hidden_size = 64 # to experiment with

run_recurrent = False    # else run Token-wise MLP
use_RNN = True          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50
check_on_added_words = False # to check on the my_text_array in loader
# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)

        # what else?
        self.in2output = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        combined = torch.cat((x, hidden_state), 1)
        h = self.sigmoid(self.in2hidden(combined))
        y = self.sigmoid(self.in2output(h))
        return y, h

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

        self.z_weights = nn.Linear(hidden_size + input_size, hidden_size)
        self.r_weights = nn.Linear(hidden_size + input_size, hidden_size)
        self.weights = nn.Linear(hidden_size + input_size, hidden_size)
        self.output_weights = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # Implementation of GRU cell
        combined = torch.cat((x, hidden_state), 1)  # TODO check correct cat

        z = self.sigmoid(self.z_weights(torch.cat((hidden_state, x), 1)))
        r = self.sigmoid(self.r_weights(torch.cat((hidden_state, x), 1)))

        h_bar = self.tanh(self.weights(torch.cat((r * hidden_state, x), 1)))
        h = (1 - z) * hidden_state + z * h_bar
        output = self.output_weights(h)
        return output, h

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.layers = nn.Sequential(
            MatMul(input_size, round(0.7*input_size)),
            nn.ReLU(),
            MatMul(round(0.7*input_size), round(0.3*input_size)),
            nn.ReLU(),
            MatMul(round(0.3 * input_size), round(output_size)),
        )

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        x = self.layers(x)
        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        # rest ...
        # self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        # query = ...
        # keys = ...
        # vals = ...

        return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each word obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    for i in range(len(rev_text)):
        print(f"{rev_text[i]} ({round(float(sbs1[i]),2)}, {round(float(sbs2[i]),2)})", end=", ")
    print()
    sbs1_avg, sbs2_avg = np.mean(sbs1), np.mean(sbs2)
    predictions = torch.softmax(torch.tensor([sbs1_avg, sbs2_avg]), dim=0)
    print(f"Prediction Label: ({round(predictions[0].item(),2)},{round(predictions[1].item(),2)})\n"
          f"Real Label: ({lbl1},{lbl2})")
    print()


# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)

print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))


if check_on_added_words :
    criterion = nn.CrossEntropyLoss()
    from loader import my_text

    labels, reviews, reviews_text = my_text()
    if run_recurrent:
        hidden_state = model.init_hidden(int(labels.shape[0]))
        for i in range(num_words):
            output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
    else:
        sub_score = model(reviews)
        for r in range(len(reviews_text)):
            for i in range(len(reviews_text[r])):
                print(f"{reviews_text[r][i]} ({round(float(sub_score[r][i][0]),2)},{round(float(sub_score[r][i][1]),2)}), ", end="")
            print()
        output = torch.mean(sub_score, 1)

    loss = criterion(output, labels)
    print((torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).bool())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0

# training steps in which a test step is executed every test_interval
train_accuracy, test_accuracy = [], []
array_test_indexes = np.array([])
train_accuracy_closest = []
test_loss_list = []
train_loss_list = []
train_loss_closest = []
counter = 0
for epoch in range(num_epochs):
    itr = 0  # iteration counter within each epoch

    for labels, reviews, reviews_text in train_dataset:  # getting training batches
        counter += 1
        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            array_test_indexes = np.append(array_test_indexes, counter)
            train_accuracy_closest.append(train_accuracy[-1])
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
        else:
            test_iter = False

        # Recurrent nets (RNN/GRU)

        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

        else:

            # Token-wise networks (MLP / MLP + Atten.)

            sub_score = []
            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            output = torch.mean(sub_score, 1)

        # cross-entropy loss
        loss = criterion(output, labels)

        if test_iter:
            test_accuracy.append(
                (torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).float().sum() / output.shape[0])

        else:
            train_accuracy.append(
                (torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).float().sum() / output.shape[0])

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            test_loss_list.append(test_loss)
            train_loss_closest.append(train_loss_list[-1])
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            train_loss_list.append(train_loss)
        if test_iter:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f},"
            )

            if not run_recurrent:
                nump_subs = sub_score.detach().numpy()
                labels = labels.detach().numpy()
                print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])

            # saving the model
            torch.save(model.state_dict(), model.name() + ".pth")

# print(len(train_accuracy), len(test_accuracy))
# plt.plot((array_test_indexes - 1).tolist(), train_accuracy_closest,
#          label=f'train accuracy')
# plt.plot(array_test_indexes.tolist(), test_accuracy, label=f'test accuracy')
# plt.xlabel("# batch")
# plt.title(f"Train and Test accuracy per iteration")
# plt.grid(True)
# plt.legend()
# plt.show()
#
# print(len(train_accuracy), len(test_accuracy))
# plt.plot((array_test_indexes - 1).tolist(), train_accuracy_closest,
#          label=f'train accuracy')
# plt.plot(array_test_indexes.tolist(), test_accuracy, label=f'test accuracy')
# plt.xlabel("# batch")
# plt.title(f"Train and Test accuracy per iteration")
# plt.grid(True)
# plt.legend()
# plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot train and test accuracy on the first subplot
ax1.plot((array_test_indexes - 1).tolist(), train_accuracy_closest, label='train accuracy')
ax1.plot(array_test_indexes.tolist(), test_accuracy, label='test accuracy')
ax1.set_xlabel("# batch")
ax1.set_title("Train and Test accuracy per iteration")
ax1.grid(True)
ax1.legend()

# Plot train and test loss on the second subplot
ax2.plot((array_test_indexes - 1).tolist(), train_loss_closest, label='train loss')
ax2.plot(array_test_indexes.tolist(), test_loss_list, label='test loss')
ax2.set_xlabel("# batch")
ax2.set_title("Train and Test loss per iteration")
ax2.grid(True)
ax2.legend()

# Show the plots
plt.tight_layout()
plt.show()