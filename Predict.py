import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
# Load model
model = load_model('model.h5')


def predict(txt):
    max_fatures = 100
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(txt)
    X = tokenizer.texts_to_sequences(txt)
    X = pad_sequences(X, maxlen=80)

    Y_pred = model.predict(X)

    for i in range(len(Y_pred)):
        if np.argmax(Y_pred[i]) == 0:
            print('Text:', txt[i], '\nSentiment: Komentar Cyberbulying')
        else:
            print('Text:', txt[i], '\nSentiment: Komentar Biasa')


txt = [
    'kebiasaan balajaer nyampah d ig para artis..suka2 yg punya ig lah mau bikin caption apa,kok balajaer yg heboh dan asik ceramahin yg punya ig.tar lama2 d bikinin lagu sm teh melly loh balajaer yg berjudul',
    'Kamu baik banget deh, boleh ga aku temenan sama kamu',
    'jangan tolol banget deh jadi orang, sadar diri'
       ]

predict(txt)