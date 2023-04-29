import matplotlib.pyplot as plt
import numpy
import pandas


DataPD = pandas.read_csv('airline-passengers.csv')


Data = DataPD.iloc[:, 1].values
Month = numpy.arange(1, Data.size+1)


plt.plot(Month, Data)

plt.xlabel('Month')
plt.ylabel('Number Of Passengers')
plt.title('Airline Passengers')

plt.show()
