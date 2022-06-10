import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense, LSTM


text = open('Proust.txt', encoding = 'utf-8').read().lower()
#text.decode('utf-8')

print(len(text))
print(text[400000:400500])