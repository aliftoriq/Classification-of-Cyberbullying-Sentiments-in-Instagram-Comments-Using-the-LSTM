import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
import pickle

# load the trained model
model = load_model("modelInstagram.h5")

# load the tokenizer
tokenizer = Tokenizer(num_words=2000, split=' ')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(tweet):
    # preprocess the tweet
    tweet = tweet.lower()
    tweet = re.sub('[^a-zA-z0-9\s]', '', tweet)
    # vectorize the tweet using the pre-fitted tokenizer
    tweet = tokenizer.texts_to_sequences([tweet])
    # pad the tweet to have the same shape as the model input
    tweet = pad_sequences(tweet, maxlen=114, dtype='int32', value=0)
    print(tweet)
    # predict the sentiment using the loaded model
    sentiment = model.predict(tweet, batch_size=1, verbose=2)[0]
    if np.argmax(sentiment) == 0:
        return "negative"
    elif np.argmax(sentiment) == 1:
        return "positive"

tweet = "Kakak Cantik "
sentiment = predict_sentiment(tweet)
print(sentiment)