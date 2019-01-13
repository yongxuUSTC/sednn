from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


def DNN(n_concat, n_freq):
    
    hidden_units = 2048
    
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    
    return model