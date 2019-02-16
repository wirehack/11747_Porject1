from collections import defaultdict
import numpy as np

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            x.append(np.array([w2i[x] for x in words.split(" ")]))
            y.append(t2i[tag])
    return np.array(x), np.array(y)


read_dataset("topicclass/topicclass_train.txt")
read_dataset("topicclass/topicclass_valid.txt")
i = 0
i2w = {v: k for k, v in w2i.items()}
word2vecline = dict()
with open('crawl-300d-2M-subword.vec') as f:
    f.readline()
    for line in f:
        line_sp = line.split()
        if line_sp[0] in w2i:
            word2vecline[line_sp[0]] = line

with open('pretrain-emb.txt', 'w') as f:
    f.write('{} 300\n'.format(len(w2i)))
    f.write('<unk> {}\n'.format(' '.join(list(map(str, np.random.rand(300))))))
    for i in range(1, len(w2i)):
        if i2w[i] in word2vecline:
            f.write(word2vecline[i2w[i]])
        else:
            f.write('{} {}\n'.format(i2w[i], ' '.join(list(map(str, np.random.rand(300))))))
            print(i2w[i], i)
