from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


class StockPredictionNetwork:
    def __init__(self, inputShape):
        self.inputShape = inputShape

    def getNetwork(self):

        # create a model
        model = Sequential()

        # first LSTM layer and a Dropout regularization
        model.add(LSTM(units= 50, return_sequences= True, input_shape= self.inputShape))
        model.add(Dropout(0.2))

        # second LSTM layer and Dropout regularization
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        # third LSTM layer and Dropout regularization
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        # fourth LSTM layer and Dropout regularization
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(units=1))

        return model