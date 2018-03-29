from WordsEmbedding import  *
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from keras.layers import LSTM
#from keras.layers import Dense
#from keras.models import Sequential
import fasttext as ft


def random_forest():
    rfc = RandomForestClassifier(n_estimators= 100)
    rfc = rfc.fit(train_embed, train_labels)
    rfc_predicted = rfc.predict(test_embed)
    print(accuracy_score(test_labels, rfc_predicted))

def svm():
    svc_clf = SVC()
    svc_clf = svc_clf.fit(train_embed, train_labels)
    svc_predicted = svc_clf.predict(test_embed)
    print(accuracy_score(test_labels, svc_predicted))

def naive_bayes():
    clf = GaussianNB().fit(train_embed, train_labels)
    nb_predicted = clf.predict(test_embed)
    print(accuracy_score(test_labels, nb_predicted))

def knn():
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_embed, train_labels)
    knn_predicted = neigh.predict(test_embed)
    print(accuracy_score(test_labels, knn_predicted))

# def FastText():
#     #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
#     #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
#     ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = 300)
#     print(len(ftmodel['langgg']))
#     print(ftmodel.words)
#     # ftpredicted = []
#     # for text in test_data:
#     #     lb = ftmodel.predict(text)
#     #     ftpredicted.append(int(lb[0][0]))
#     # print(accuracy_score(test_labels, ftpredicted))
#     #labels = ftmodel.predict(text)


def fit_lstm():
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_embed.shape[1])))
    #model.add(SimpleRNN(50, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_embed, train_labels, epochs=1, batch_size= 32, validation_data=(val_embed, val_labels), verbose=2,
                        shuffle=False)
    lstm_predicted = model.predict(test_embed)
    print("lstm")
    print(accuracy_score(test_labels, lstm_predicted))
    return model, history


train_embed, train_labels, test_embed, test_labels = FastText()
naive_bayes()
# random_forest()
# svm()
# knn()
#FastText()
#fit_lstm()
