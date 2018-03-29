from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from PreprocessData import *
import numpy as np
import nltk
import fasttext as ft
import requests


def separate(string):
    tokenized = nltk.word_tokenize(string)
    #tk = requests.post('http://192.168.23.86:2673/coreml/tokenizer', data = {'text': string, 'body': 'APPLICATION_FORM_URLENCODED'})
    #print(tk.text)
    #print(type(tk))
    return tokenized#tk.text.split(" ")

def tf_idf():
    vectorizer = TfidfVectorizer(analyzer=separate)
    vectorizer = vectorizer.fit(train_data)
    # #print(vectorizer)
    idf = vectorizer.idf_
    #print(dict(zip(vectorizer.get_feature_names(), idf)))
    #
    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    print(train_embed)
    return train_embed, train_labels, test_embed, test_labels

def bag_of_word():
    vectorizer = CountVectorizer(analyzer=separate)
    vectorizer = vectorizer.fit(train_data)

    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    # print(train_embed)

    return train_embed, train_labels, test_embed, test_labels

def n_gram():
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3,3))
    vectorizer = vectorizer.fit(train_data)

    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    # print(train_embed)

    return train_embed, train_labels, test_embed, test_labels

def FastText():
    dimens = 300
    #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
    #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
    ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = dimens)
    # print(len(ftmodel['langgg']))
    # print(ftmodel.words)
    train_embed = []
    test_embed = []

    for text in train_data:
        tokens = nltk.word_tokenize(text)
        embed = []
        for i in range(dimens):
            embed.append(0)
        for token in tokens:
            vec = ftmodel[token]
            for i in range(dimens):
                embed[i] += vec[i]
        for i in range(dimens):
            embed[i] = embed[i]/(len(tokens))
        train_embed.append(embed)
        #print(embed)

    for text in test_data:
        tokens = nltk.word_tokenize(text)
        embed = []
        for i in range(dimens):
            embed.append(0)
        for token in tokens:
            vec = ftmodel[token]
            for i in range(dimens):
                embed[i] += vec[i]
        for i in range(dimens):
            embed[i] = embed[i] / (len(tokens))
        test_embed.append(embed)


    train_embed = np.array(train_embed)
    test_embed  = np.array(test_embed)
    # ftpredicted = []
    # for text in test_data:
    #     lb = ftmodel.predict(text)
    #     ftpredicted.append(int(lb[0][0]))
    # print(accuracy_score(test_labels, ftpredicted))
    #labels = ftmodel.predict(text)
    return train_embed, train_labels, test_embed, test_labels


def merge(train_data, train_labels, val_data, val_label):
    data = []
    labels = []
    for i in range(len(train_data)):
        data.append(train_data[i])
        labels.append(train_labels[i])
    for i in range(len(val_data)):
        data.append(val_data[i])
        labels.append(val_labels[i])
    return data,labels

train_data, train_labels = convert_to_array("data/train.txt")
test_data, test_labels = convert_to_array("data/test.txt")
val_data, val_labels = convert_to_array("data/val.txt")
train_data, train_labels = merge(train_data, train_labels, val_data, val_labels)
#FastText()
#tf_idf()
