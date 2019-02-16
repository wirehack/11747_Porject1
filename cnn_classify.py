from collections import defaultdict
import time
import random
import torch
import numpy as np


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags, pretrain_emb):
        super(CNNclass, self).__init__()
        """ layers """
        self.embedding = torch.nn.Embedding.from_pretrained(pretrain_emb, freeze=True)
        # Conv 1d
        self.conv_1ds = torch.nn.ModuleList()
        for i in range(len(num_filters)):
            self.conv_1ds.append(torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters[i],
                                                 kernel_size=window_size[i], stride=1, padding=0, bias=True))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.projection_layer = torch.nn.Linear(in_features=sum(num_filters), out_features=ntags, bias=True)

    def forward(self, words):
        emb = self.embedding(words)  # nwords x emb_size
        emb = emb.permute(0, 2, 1)  # batch x emb_size x nwords
        h_list = []
        for conv_1d in self.conv_1ds:
            h = conv_1d(emb)  # batch x num_filters[i] x nwords
            # Do max pooling
            h = h.max(dim=2)[0]  # batch x num_filters[i]
            h_list.append(h)
        h = torch.cat(h_list, dim=1)
        h = self.relu(h)
        h = self.dropout(h)
        out = self.projection_layer(h)  # size(out) = 1 x ntags
        return out


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename, test=False):
    x = []
    y = []
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            x.append(np.array([w2i[x] for x in words.split(" ")]))
            if not test:
                y.append(t2i[tag])
    return np.array(x), np.array(y)


def load_pretrain_embeddings(filename):
    embeddings = []
    with open(filename) as f:
        f.readline()
        for line in f:
            line_sp = line.rstrip().split(" ")
            emb = list(map(float, line_sp[1:]))
            embeddings.append(emb)
    return torch.Tensor(embeddings)


# Read in the data
trainx, trainy = read_dataset("topicclass/topicclass_train.txt")
devx, devy = read_dataset("topicclass/topicclass_valid.txt")
w2i = defaultdict(lambda: UNK, w2i)
nwords = len(w2i)
ntags = len(t2i)
testx, _ = read_dataset("topicclass/topicclass_test.txt")

# Define the model
EMB_SIZE = 300
WIN_SIZE = [2, 3, 4, 5, 6]
FILTER_SIZE = [100, 100, 100, 100, 100]
batch_size = 256
pretrain_emb = load_pretrain_embeddings('pretrain-emb.txt')

# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags, pretrain_emb)
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()
print("start training")


def pad_pattern_end(x):
    result = []
    max_len = max(max([instance.shape[0] for instance in x]), max(WIN_SIZE))
    for instance in x:
        result.append(np.pad(instance, (0, max_len - instance.shape[0]), 'constant', constant_values=0))
    return np.stack(result)


def write_predictions(filename, predictions):
    predictions = list(map(int, predictions.cpu()))
    i2t = {k: v for (v, k) in t2i.items()}
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write("{}\n".format(i2t[pred]))


idxs = np.arange(len(trainx))
best_acc = 0

for ITER in range(100):
    # Perform training
    random.shuffle(idxs)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    model.train()
    for b in range(0, idxs.shape[0], batch_size):
        batchx = pad_pattern_end(trainx[idxs[b:b + batch_size]])
        batchy = trainy[idxs[b:b + batch_size]]
        words_tensor = torch.tensor(batchx).type(type)
        tag_tensor = torch.tensor(batchy).type(type)
        outputs = model(words_tensor)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == tag_tensor).sum().item()
        my_loss = criterion(outputs, tag_tensor)
        train_loss += my_loss.item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss * batch_size / idxs.shape[0], train_correct / idxs.shape[0], time.time() - start))
    # Perform testing
    dev_correct = 0.0
    model.eval()
    dev_predictions = []
    for b in range(0, len(devx), batch_size):
        batchx = pad_pattern_end(devx[b:b + batch_size])
        batchy = devy[b:b + batch_size]
        words_tensor = torch.tensor(batchx).type(type)
        tag_tensor = torch.tensor(batchy).type(type)
        outputs = model(words_tensor)
        _, predicted = torch.max(outputs.data, 1)
        dev_correct += (predicted == tag_tensor).sum().item()
        dev_predictions.append(predicted)
    dev_predictions = torch.cat(dev_predictions, dim=0)
    dev_acc = dev_correct / len(devx)
    print("iter %r: test acc=%.4f" % (ITER, dev_acc))
    if dev_acc > best_acc:
        best_acc = dev_acc
        print("best acc: %.4f" % best_acc)
        write_predictions("valid_predictions.txt", dev_predictions)
        test_predictions = []
        for b in range(0, len(testx), batch_size):
            batchx = pad_pattern_end(testx[b:b + batch_size])
            words_tensor = torch.tensor(batchx).type(type)
            outputs = model(words_tensor)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.append(predicted)
        test_predictions = torch.cat(test_predictions, dim=0)
        write_predictions("test_predictions.txt", test_predictions)
