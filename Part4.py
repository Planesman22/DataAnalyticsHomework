import pandas
import numpy
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense

WindowSize = 3

numpy.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DataPD = pandas.read_csv('airline-passengers.csv')

Data = DataPD.iloc[:, 1].values.astype('float')

# Nah good mental excersize, normalize!
Max = 0
for I in range(len(Data)):
    if Data[I] > Data[Max]:
        Max = I
Scale = Data[Max]
Data = Data/Scale

# We can just manually extract 67% of it
TrainData = Data[:int(len(Data)*0.67)]
TestData = Data[int(len(Data)*0.67):]

def getSequence(Data, WindowSize):
    X, Y = [], []
    for i in range(len(Data) - WindowSize):
        X.append(Data[i:(i + WindowSize)])
        Y.append(Data[i + WindowSize])
    return numpy.array(X).reshape(-1, WindowSize), numpy.array(Y).reshape(-1, 1)

TrainX, TrainY = getSequence(TrainData, WindowSize)
TestX, TestY = getSequence(TestData, WindowSize)

SeqModel = Sequential([
    LSTM(4, input_shape=(WindowSize, 1), activation='sigmoid'),
    Dense(1)
])

SeqModel.compile(loss='mean_squared_error', optimizer='adam')
SeqModel.fit(TrainX, TrainY, epochs=100, batch_size=1, verbose=2)

trainPredict = SeqModel.predict(TrainX)
testPredict = SeqModel.predict(TestX)

plt.plot(Data*Scale, label='Actual')
plt.plot(trainPredict*Scale, label='Train')

testPredict = numpy.insert(testPredict, 0 ,numpy.zeros(trainPredict.shape[0]))

plt.plot(testPredict*Scale, label='Test')
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.title("Passenger Prediction")
plt.legend()
plt.show()
