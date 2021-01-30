import warnings,logging,os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import nltk
from nltk.stem.lancaster import LancasterStemmer
import random,json,tflearn,pickle
import numpy as np

from utils.v1_utils import *
from config.global_vars import *

# * Read intents file
with open('config/intents.json') as f:
    intents = json.load(f)

# * Loading the data req for building the model
with open('model/v1/data.pickle','rb') as f:
    vocab,labels,training,output = pickle.load(f)

# * Initializing the model
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# * load trained model wts
model.load('model/v1/chabot_v1.tflearn')

# * text colors
class txtcolor:
    YOU = '\033[94m'
    LYDYA_SUCCESS = '\033[92m'
    LYDYA_FAIL = '\033[91m'
    
    
def chat():
    print("start talking with the bot!")
    while True:
        inp = input(f"{txtcolor.YOU}you: ")
        results = model.predict([get_bow(inp,vocab)])
        results_index = np.argmax(results)
        results_proba = np.max(results)
        if results_proba < THRESHOLD:
            print(f'{txtcolor.LYDYA_FAIL}'+str(random.choice(FAILURE_RESP)))
        else:
            tag = labels[results_index]
            for intent in intents['intents']:
                if tag == "goodbye" and tag == intent['tag']:
                    print(f"{txtcolor.LYDYA_SUCCESS}Lydya: "+ str(random.choice(intent['responses'])))
                    return None
                elif tag == intent['tag']:
                    print(f"{txtcolor.LYDYA_SUCCESS}Lydya: "+ str(random.choice(intent['responses'])))            