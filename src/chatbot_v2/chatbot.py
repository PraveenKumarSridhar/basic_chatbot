import warnings,logging,os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random,json,os,pickle
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

from config.global_vars import *

# * Read intents file
with open('config/intents.json') as f:
    intents = json.load(f)
# * Lables get labels
labels = [intent['tag'] for intent in intents['intents']] 


# * Load tokenizer
tokenizer_path = os.path.join(TOKENIZERS,'tokenizer.pickle')
with open(tokenizer_path,'rb') as f:
    tokenizer = pickle.load(f)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# * Load trained model wts
model = load_model('model/v2/bilstm.h5')
    
maxlen =20

def chat():
    inp = input(f"{txtcolor.YOU}you: ")
    seq = pad_sequences(tokenizer.texts_to_sequences(np.array([inp])),maxlen = maxlen)
    results = model.predict(seq)
    results_index = np.argmax(results)
    results_proba = np.max(results)
    if results_proba < THRESHOLD:
        print(f'{txtcolor.LYDYA_FAIL}'+str(random.choice(FAILURE_RESP)))
        return True
    else:
        tag = labels[results_index]
        for intent in intents['intents']:
            if tag == "goodbye" and tag == intent['tag']:
                print(f"{txtcolor.LYDYA_SUCCESS}Lydya: "+ str(random.choice(intent['responses'])))
                return False
            elif tag == intent['tag']:
                print(f"{txtcolor.LYDYA_SUCCESS}Lydya: "+ str(random.choice(intent['responses'])))
                return True            