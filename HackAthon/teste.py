import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = [i for i in range(1, len(acc) + 1)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


#Parte da analise de sentimentos;
filepath_dict = {'yelp':   'yelp_labelled.txt',
                 'imdb':   'imdb_labelled.txt',
                 'text'	:	'dest.txt',
                 'fakenews': 'fakenews_dataset/fake_or_real_news.csv'}

def join_files(files_dict=filepath_dict):
    df_list = []
    for source, filepath in files_dict.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep=",")
        df['source'] = source
        df_list.append(df)

    return pd.concat(df_list)


def logistic_train_regression(sentences_train, sentences_test, label_train, label_test, source=""):
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, label_train)

    score = classifier.score(X_test, label_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))

    return classifier, vectorizer


def made_data(df):
    tmp_dict = {}
    for source in df['source'].unique():  # e esse for pra percorrer os arquivos aqui é uma maravilha tbm;
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values
        #if source=="fakenews":
            #print(sentences[4], y[:8])
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25,
                                                                            random_state=1000)

        tmp_dict[source] = ((sentences_train, y_train), (sentences_test, y_test))
    return tmp_dict



def convolutional_processing(sentences_train, sentences_test, y_train, y_test, source=""):

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 50

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embedding_dim = 100

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                         epochs=100,
                         verbose=False,
                         validation_data=(X_test, y_test),
                         batch_size=5)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

def save_model(model,filename,extention="sav"):
    with open(filename+"."+extention, "wb") as f:
        pickle.dump(model,f)

def load_model(filename, extention="sav"):
    tmp = None
    with open(filename+"."+extention,"rb") as f:
        tmp = pickle.load(f)
    return tmp

if __name__ == "__main__":
    dict_files = join_files()
    (sentences_train, y_train), (sentences_test, y_test) = made_data(dict_files)["fakenews"]
    log_reg_name, vec_name = "logistic_regression", "vectorize"
    model = load_model(log_reg_name)
    vectorizer = load_model(vec_name)
    if not model and not vectorizer:
        model, vetorizer = logistic_train_regression(sentences_train, sentences_test, y_train, y_test, source="fakenews")


    #convolutional_processing(sentences_train, sentences_test, y_train, y_test)
