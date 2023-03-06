
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('dataset_komentar_instagram_cyberbullying.csv')
# Keeping only the neccessary columns
data = data[['Instagram Comment Text','Sentiment']]

data['Instagram Comment Text'] = data['Instagram Comment Text'].apply(lambda x: x.lower())
data['Instagram Comment Text'] = data['Instagram Comment Text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

negative = data[data['Sentiment'] == 'positive']
positive = data[data['Sentiment'] == 'negative']
print("= = = Dataset = = =")
print("data positive (200 x 2)\t : ", positive.size)
print("data negative (200 x 2)\t : ", negative.size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['Instagram Comment Text'].values)
X = tokenizer.texts_to_sequences(data['Instagram Comment Text'].values)
X = pad_sequences(X)
# save the tokenizer object to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
history = model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

validation_size = 20

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

# pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
# for x in range(len(X_validate)):

#     result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

#     if np.argmax(result) == np.argmax(Y_validate[x]):
#         if np.argmax(Y_validate[x]) == 0:
#             neg_correct += 1
#         else:
#             pos_correct += 1

#     if np.argmax(Y_validate[x]) == 0:
#         neg_cnt += 1
#     else:
#         pos_cnt += 1

# print("pos_acc", pos_correct / pos_cnt * 100, "%")
# print("neg_acc", neg_correct / neg_cnt * 100, "%")
model.save("modelInstagram.h5")
