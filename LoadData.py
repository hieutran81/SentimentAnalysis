import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename, delimiter="\n")
    data = df.values[:,0]
    texts = []
    labels = []
    for i in range(data.shape[0]):
        seq = data[i][5:]
        # print(data[i])
        # print(i)
        label = int(data[i][0])-1
        texts.append(seq)
        labels.append(label)
    return texts, np.array(labels)



# load_data("data/test.txt")

