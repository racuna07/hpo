import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataUtils(object):
    def __init__(self):
        # We read the data from the csv file
        self.dfData = pd.read_csv('C:\\Users\\Rodrigo\\PycharmProjects\\hpo\\resources\\RawDataWithRainTransformed.csv', sep=',')

        # We generate the classes and features of the data
        self.le = LabelEncoder()
        self.le.fit(self.dfData['Result'])
        self.allClasses = self.le.transform(self.dfData['Result'])
        self.allFeatures = self.dfData.drop(['Result'], axis=1)

        # We create the training, testing and validation sets randomizing the order of the data set
        self.x_trainAndTest, self.x_validation, self.y_trainAndTest, self.y_validation = \
            train_test_split(self.allFeatures, self.allClasses, test_size=0.20, random_state=42)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_trainAndTest,
                                                                                self.y_trainAndTest,
                                                                                test_size=0.20,
                                                                                random_state=42)
        # We scale the data
        self.scaler = StandardScaler()
        self.scaler.fit(self.x_train)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def get_training_data(self):
        return self.x_train, self.y_train

    def get_testing_data(self):
        return self.x_test, self.y_test
