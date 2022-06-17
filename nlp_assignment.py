import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import numpy as np
import os
import tensorflow as tf
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.io import savemat
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def preprocess(filename):
    # function to preprocess a given file and output a list of words in its content.
    
    with open(filename) as f:
        lines = f.readlines()
    
    lines_clean = []
    
    # remove sentences with these characters which are likely to be metadata
    rem_chars = ['--', ':', '@', '<', '>']
    for i in range (len(lines)):
        if any(val in lines[i] for val in rem_chars) and lines[i] != '\n':
            lines_clean.append(lines[i])
        # save a maximum of 30 lines
        if len(lines_clean) >=20:
            break
    lines_clean = ''.join(lines_clean)
    
    # tokenize the mail, remove punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(lines_clean)
    
    # remove common stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # lemmatise the words, remove any character other than letters, and convert to lower case
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = []
    for token in tokens:
        if token not in set(stopwords):
            token = re.sub('[^a-zA-Z]', '', token)
            if token != '':
                lemmas.append(lemmatizer.lemmatize(token.lower()))
    return lemmas

def model(embedding_matrix, num_tokens, embedding_dim, max_seq):
    # Returns the LSTM model
    inputs = tf.keras.layers.Input(shape=[max_seq], name='Input')
    x = tf.keras.layers.Embedding(
                num_tokens,
                embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=True,
                mask_zero=True,
                input_length=max_seq,
                name='Word_Embedding'
            )(inputs)
    x = tf.keras.layers.LSTM(100, return_sequences=True, dropout=0.4, name='LSTM_1')(x)
    x = tf.keras.layers.LSTM(100, return_sequences=False, dropout=0.4, name='LSTM_2')(x)
    x = tf.keras.layers.Dropout(0.4, name='Dropout')(x)
    x = tf.keras.layers.Dense(50, activation='elu', name='Dense')(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax', name='Softmax_Output')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def one_hot(y):
    # Returns one-hot encoding of a vector
    y1 = np.zeros((len(y), np.max(y)+1))
    for i in range(len(y)):
        y1[i, y[i]] = 1
    return y1

def get_metrics(clf, x, y_true):
    y_pred = clf.predict(x)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    y_true_hot = one_hot(y_true)
    y_pred_hot = one_hot(y_pred)
    auroc = roc_auc_score(y_true_hot, y_pred_hot, multi_class='ovr', average='micro')
    return accuracy, auroc, precision, recall

tf.random.set_seed(42)
np.random.seed(42)
          
num_folders = 8

x = []
y = []

# read all files and preprocess them
for i in range (num_folders):
    files = os.listdir('./enron_with_categories/' + str(i+1) + '/')
    for file in files:
        if file.endswith('.cats'):
            with open('./enron_with_categories/' + str(i+1) + '/' + file) as f:
                lines = f.readlines()
            lines = lines[0].split(',')
            if lines[0] == '1':
                if int(lines[1]) <= 6:
                    file = file.split('.')
                    x.append(preprocess('./enron_with_categories/' + str(i+1) + '/' + file[0] + '.txt'))
                    y.append(int(lines[1])-1)

y = np.array(y)
y_int = y.copy()
y = one_hot(y)

# find the vocabulary, only keep top 5000 words
num_tokens = 5000
temp = [' '.join(x1) for x1 in x]
vectorizer = CountVectorizer()
cv_fit = vectorizer.fit_transform(temp)
vocabulary = vectorizer.vocabulary_
freq_sort = np.argsort(-1 * cv_fit.toarray().sum(axis=0))
for word in list(vocabulary):
    if freq_sort[vocabulary[word]] >= num_tokens:
        del vocabulary[word]
for i, word in enumerate(vocabulary):
    vocabulary[word] = i

# load embeddings from pretrained model                    
embeddings = {}
embedding_dim = 50
with open('./Pre-trained Embeddings/glove.6B.50d.txt', encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings[word] = coefs

# create an embedding matrix
# Randomly initialize for words not in the pre-trained model
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for i, word in enumerate(vocabulary):
    if word in embeddings:
        embedding_matrix[i,:] = embeddings[word]
    else:
        embedding_matrix[i,:] = np.random.rand(embedding_dim)

# convert word data to integers and pad with zeros
data = np.zeros((len(x), max([len(i) for i in x])))     
for i, sent in enumerate(x):
    j = 0
    for word in sent:
        if word in vocabulary:
            data[i,j] = vocabulary[word] + 1
            j += 1

# tf-idf features
counts = np.zeros((data.shape[0], num_tokens))
for i, sent in enumerate(x):
    for j, word in enumerate(vocabulary):
        counts[i,j] = x[i].count(word)
tf_idf_data = TfidfTransformer().fit_transform(counts).toarray()

data = data[:,~np.all(data == 0, axis=0)]
y = y[~np.all(data == 0, axis=1)]
tf_idf_data = tf_idf_data[~np.all(data == 0, axis=1)]
data = data[~np.all(data == 0, axis=1)]

# 80-20 train-test split stratified on the labels
X_train, X_test, y_train, y_test, X_tf_idf_train, X_tf_idf_test, y_int_train, y_int_test = train_test_split(
    data, y, tf_idf_data, y_int, test_size=0.20, random_state=42, stratify=y)

# svm-classifier with tf_idf features
clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_tf_idf_train, y_int_train)
svm_metrics = {}
svm_metrics['train_accuracy'],  svm_metrics['train_auroc'],\
 svm_metrics['train_precision'], svm_metrics['train_recall'] = get_metrics(clf, X_tf_idf_train, y_int_train)
svm_metrics['test_accuracy'],  svm_metrics['test_auroc'],\
 svm_metrics['test_precision'], svm_metrics['test_recall'] = get_metrics(clf, X_tf_idf_test, y_int_test)
 
# load, compile and train the model
lstm_model = model(embedding_matrix, num_tokens, embedding_dim, data.shape[1])
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(), 
                    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
history = lstm_model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=[X_test, y_test])

# save the model and the history of the metrics and losses
lstm_model.save('./Saved Model/lstm_model.h5')
savemat('./Saved Metrics/lstm_model_history.mat', history.history)
savemat('./Saved Metrics/svm_metrics.mat', svm_metrics)