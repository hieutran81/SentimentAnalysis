from WordsEmbedding import  *
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lime import lime_text
from lime.lime_text import  LimeTextExplainer
from sklearn.pipeline import make_pipeline
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Embedding
# from keras.layers import Flatten
# from keras.preprocessing import sequence
# from keras.layers import LSTMtext
import fasttext as ft
import pickle


def random_forest():
    trees = [100,500, 1000, 3000, 5000, 10000]
    for num in trees:
        rfc = RandomForestClassifier(n_estimators= num)
        print("start fit")
        rfc = rfc.fit(train_embed, train_labels)
        print("finish fit")
        # with open("model/random_forest.pk","wb") as f:
        #     pickle.dump(rfc,f)
        # with open("model/vectorizer.pk","wb") as f:
        #     pickle.dump(vectorizer,f)
        print("finish dump")
        rfc_predicted = rfc.predict(test_embed)
        print(accuracy_score(test_labels, rfc_predicted))

def explain_result():
    class_names = ['positive', 'negative', 'neutral']
    with open("model/random_forest.pk", "rb") as f:
        rf = pickle.load(f)
    c = make_pipeline(vectorizer, rf)

    explainer = LimeTextExplainer(class_names=class_names)
    text = "Không "
    print(c.predict_proba([text]))
    exp = explainer.explain_instance(text, c.predict_proba, num_features=6)
    print(exp.as_list())


def predict_sentiment(text):
    with open("model/random_forest.pk", "rb") as f:
        rfc = pickle.load(f)
    with open("model/vectorizer.pk", "rb") as f:
        vectorizer = pickle.load(f)
    c = make_pipeline(vectorizer, rfc)
    return rfc.predict(vectorizer.transform([text]))[0],c.predict_proba([text])

def svm():
    cp = [1.0]
    for c in cp:
        svc_clf = SVC(C = c)
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

max_words = 80

# train_embed, train_labels, test_embed, test_labels = fasttext(maxwords = max_words)
# # #explain_result()
# random_forest()
# #svm()
# # knn()
# #FastText()
# #fit_lstm()
