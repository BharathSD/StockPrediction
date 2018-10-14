from Data.LoadData import GoogleStockData as stockdata
import numpy as np
from Network.StockPredictionNetwork import StockPredictionNetwork
import configparser
import os
import matplotlib
matplotlib.use('QT4agg')
import matplotlib.pyplot as plt

def parseConfigurations(configPath:str):
    print('parsing configfile...')
    cfg = configparser.ConfigParser()
    cfg.read(configPath)
    return cfg

class TrainigParams:
    def __init__(self, batch_size:int, epochs:int):
        self.batch_size = batch_size
        self.epochs = epochs

class Application:
    def __init__(self, model_path:str, trainingParams:TrainigParams):
        self.model_path = model_path

        # load training data
        TrainData = stockdata('Google_stock_Price_Train.csv')
        trainingData = TrainData.getData()
        self.x_train, self.y_train = self.prepareData(trainingData)

        # load testing data
        TestData = stockdata('Google_stock_Price_Test.csv')
        testingData = TestData.getData()
        self.x_test, self.y_test = self.prepareData(testingData)

        network = StockPredictionNetwork(inputShape=(self.x_train.shape[1], 1))
        self.model = network.getNetwork()

        # compiling the sequential model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # update training Parameters
        self.trainingParams = trainingParams

    def prepareData(self, datapoints: list, timesteps: int = 60):
        x_data = list()
        y_data = list()

        for i in range(timesteps, len(datapoints)):
            x_data.append(datapoints[i - timesteps:i, 0])
            y_data.append(datapoints[i, 0])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Reshape the data
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

        return x_data, y_data


    def train(self):
        history = self.model.fit(self.x_train, self.y_train,
                                 batch_size=self.trainingParams.batch_size,
                                 epochs=self.trainingParams.epochs,
                                 verbose=2,
                                 validation_data=(self.x_test, self.y_test))

        # plotting the metrics
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        plt.tight_layout()

        fig.show()
        fig.savefig('TrainingParams.png')

        # saving the model
        self.model.save(self.model_path)
        print('Saved trained model at %s ' % self.model_path)


    def inference(self):
        pass

if __name__ == '__main__':
    cfg = parseConfigurations(r'Configuration.ini')
    TrainigParamsI = TrainigParams(batch_size=int(cfg['TrainingParams']['batch_size']),
                                   epochs=int(cfg['TrainingParams']['epochs']))
    ModelFilePath = cfg['Model']['ModelPath']
    ModelFileName = cfg['Model']['ModelName']
    ModelFile = ModelFileName
    if os.path.exists(ModelFilePath):
        ModelFile = os.path.join(ModelFilePath, ModelFileName)

    modelFilePath = ModelFile
    app = Application(model_path=modelFilePath, trainingParams=TrainigParamsI)

    app.train()