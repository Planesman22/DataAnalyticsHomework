import pandas
import numpy
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense

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

# Hoodini Magic
TrainX = TrainData[:-1]
TrainY = TrainData[1:]

TestX = TestData[:-1]
TestY = TestData[1:]

# Re-Shaping, almost did a def for this but i just wanted to be fast
TrainX = numpy.vstack((TrainX, numpy.arange(1, TrainX.size+1)))
TestX = numpy.vstack((TestX, numpy.arange(1, TestX.size+1)))


TrainX = numpy.transpose(TrainX)
TestX = numpy.transpose(TestX)

TrainX = numpy.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = numpy.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))

SeqModel = Sequential([
    LSTM(4, input_shape=(1, 2), activation='sigmoid'),
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
