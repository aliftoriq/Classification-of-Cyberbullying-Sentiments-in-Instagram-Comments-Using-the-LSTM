import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Load model
model = load_model('model.h5')

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    # vectorizing the tweet by the pre-fitted tokenizer instance
    text = tokenizer.texts_to_sequences(text)
    # padding the tweet to have exactly the same shape as `embedding_2` input
    text = pad_sequences(text, maxlen=114, dtype='int32', value=0)
    print(text)
    sentiment = model.predict(text, batch_size=1, verbose=2)[0]
    if (np.argmax(sentiment) == 0):
        print("negative")
    elif (np.argmax(sentiment) == 1):
        print("positive")

predict_class('kamu goblok bgt sih, aku suka kamu padahal')