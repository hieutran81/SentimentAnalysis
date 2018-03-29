from WordsEmbedding import  *
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers import LSTM
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


def convert_one_hot(y_train, y_test):
    lab = np.amax(y_train)+1
    train_enc = np.zeros((y_train.shape[0],lab))
    train_enc[np.arange(y_train.shape[0]),y_train] = 1
    test_enc = np.zeros((y_test.shape[0], lab))
    test_enc[np.arange(y_test.shape[0]), y_test] = 1
    print(train_enc)
    print(train_enc.shape)
    print(test_enc)
    print(test_enc.shape)
    return train_enc, test_enc


def fit_lstm():
    X_train = train_embed
    y_train = np.array(train_labels)
    X_test = test_embed
    y_test = test_labels
    X_train = sequence.pad_sequences(X_train, maxlen= max_words)
    X_test  = sequence.pad_sequences(X_test, maxlen = max_words)
    y_train, y_test = convert_one_hot(y_train, y_test)
    # create the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(4, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 50, batch_size= 32, verbose=2, shuffle=False)
    score = model.evaluate(X_test, y_test, verbose = 0)
    print("Accuracy : %.2f%%" %(score[1]*100))

max_words = 50

train_embed, train_labels, test_embed, test_labels = fasttext_lstm()
fit_lstm()
# random_forest()
# svm()
# knn()
#FastText()
#fit_lstm()
