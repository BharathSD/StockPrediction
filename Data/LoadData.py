import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class GoogleStockData:
    def __init__(self, fileName, featureScaling=True):
        self.fileName = fileName
        self.featureScaling =featureScaling

    def getData(self):
        dataset = pd.read_csv(self.fileName)
        data = dataset.iloc[:,1:2].values
        #perform feature scaling
        if self.featureScaling is True:
            sc = MinMaxScaler(feature_range=(0, 1))
            data = sc.fit_transform(data)
        return data