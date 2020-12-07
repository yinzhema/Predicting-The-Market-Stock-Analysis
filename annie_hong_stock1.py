#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:04:25 2020

@author: annie
"""
import yfinance as yf
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import zero_one_loss


# fix random seed for reproducibility
np.random.seed(7)

def create_dataset(modelType, dataset, rawdataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		if (modelType == "SVM") or (modelType == "LSTMC") or (modelType == "CNN"):
			if (rawdataset[i + look_back, 0] > 0):
				dataY.append(1)
			else:
				dataY.append(0)
		else:
			dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
#end create_dataset

def create_sample( modeType, sname="AAPL", look_back=20, start="2000-01-01", end="2020-09-30" ):

	apple = yf.Ticker(sname)
	hist = apple.history(start=start, end="2020-09-30", actions=False)

	# explore dataset
	hist.head()
	#print(hist.dtypes)


	# normalize data, c reate train set and test set
	data = np.array(hist)[:,3]
	data = np.reshape(data, (-1, 1))

	ddata = np.empty_like(data)
	ddata[1:-1] = (data[1:-1] - data[0:-2]) / data[0:-2]
	ddata[0] = 0;
	ddata = ddata * 100;

	train_size = int(len(data) * 0.90)
	test_size = len(data) - train_size
	rawtrain, rawtest = ddata[0:train_size,:], ddata[train_size:len(ddata),:]
	train, test = ddata[0:train_size,:], ddata[train_size:len(ddata),:]

	scaler = MinMaxScaler(feature_range=(-1, 1))
	train = scaler.fit_transform(train)
	test = scaler.transform(test)

	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset( modelType, train, rawtrain, look_back)
	testX, testY = create_dataset( modelType, test, rawtest, look_back)

	return trainX, trainY, testX, testY, data, ddata, scaler,
#end create_sample

def createModel ( modelType = 'LSTM', seqSize=1, featureSize=1 ):
	if (modelType == 'LSTM') :
		# create and fit the LSTM network
		model = Sequential(name='LSTM')
		model.add(LSTM(20, input_shape=(seqSize, featureSize)))
		model.add(Dropout(0.2))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
	elif (modelType == 'LSTMC') :
		# create and fit the LSTM network
		model = Sequential(name='LSTMC')
		model.add(LSTM(20, input_shape=(seqSize, featureSize)))
		model.add(Dropout(0.2))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(2, activation='softmax'))
		model.compile(loss='mean_squared_error', optimizer='adam')
	elif (modelType == 'CNN') :
		# create and fit the LSTM network
		model = Sequential(name='CNN')
		model.add(Conv1D(filters=20, kernel_size=3, activation='relu', input_shape=(seqSize, featureSize)))
		model.add(Dropout(0.2))
		model.add(BatchNormalization())
		model.add(Conv1D(filters=10, kernel_size=5, activation='relu'))
		#model.add(BatchNormalization())
		#model.add(Conv1D(filters=20, kernel_size=3, activation='relu'))
		model.add(BatchNormalization())
		model.add(Flatten())
		model.add(Dense(10, activation='relu'))
		model.add(Dense(2, activation='softmax'))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.summary()
	elif (modelType == 'SVR') :
		svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
		svr_lin = SVR(kernel='linear', C=100, gamma='auto')
		svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
					   coef0=1)
		model = svr_lin
	elif (modelType == 'SVM') :
		svm_rbf = SVC(kernel='rbf', C=100, gamma=0.1 )
		svm_lin = SVC(kernel='linear', C=100, gamma='auto')
		svm_poly = SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1)

		model = svm_rbf


	return model
# end createModel

def trainModel ( modelType, model, trainX, trainY ):
	if (modelType == 'LSTM') :
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		trainY2 = np.reshape(trainY, (trainY.shape[0]))
		model.fit(trainX, trainY2, epochs=10, batch_size=10, verbose=2)
	elif (modelType == 'LSTMC') :
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		trainY2 = utils.to_categorical(trainY, 2)
		model.fit(trainX, trainY2, epochs=10, batch_size=10, verbose=2)
	elif (modelType == 'CNN') :
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		trainY2 = utils.to_categorical(trainY, 2)
		model.fit(trainX, trainY2, epochs=10, batch_size=10, verbose=2)
	elif (modelType == 'SVR') :
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
		trainY2 = np.reshape(trainY, (trainY.shape[0]))
		model.fit(trainX, trainY2)
	elif (modelType == 'SVM') :
		trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
		trainY2 = np.reshape(trainY, (trainY.shape[0]))
		model.fit(trainX, trainY2)

	return model
# end trainModel

def testModel ( modelType, model, X ):
	if (modelType == 'LSTM'):
		# reshape input to be [samples, time steps, features]
		X = np.reshape(X, (X.shape[0], X.shape[1], 1))
		Pr = model.predict(X)
	elif (modelType == 'LSTMC'):
		# reshape input to be [samples, time steps, features]
		X = np.reshape(X, (X.shape[0], X.shape[1], 1))
		Pr = model.predict(X)
		Pr = np.argmax(Pr, axis=-1)
	elif (modelType == 'CNN'):
		# reshape input to be [samples, time steps, features]
		X = np.reshape(X, (X.shape[0], X.shape[1], 1))
		Pr = model.predict(X)
		Pr = np.argmax(Pr, axis=-1)
	elif (modelType == 'SVR') :
		X = np.reshape(X, (X.shape[0], X.shape[1]))
		Pr = model.predict(X)
		Pr = np.reshape(Pr, (Pr.shape[0], 1))
	elif (modelType == 'SVM') :
		X = np.reshape(X, (X.shape[0], X.shape[1]))
		Pr = model.predict(X)
		Pr = np.reshape(Pr, (Pr.shape[0], 1))

	return Pr
#end testModel

def calcReturn ( trainPredict, testPredict, data, ddata, look_back = 20 ):
	# trainy and ddata are with offset look_back
	traPredict = np.reshape(trainPredict, len(trainPredict));
	tstPredict =  np.reshape(testPredict,len(testPredict))
	iinvest = 100; # initial invest
	rtrain = np.ones(traPredict.shape) * iinvest;
	rtest = np.ones(tstPredict.shape) * iinvest;


	for i in range(trainPredict.shape[0]):
		p = traPredict[i];
		r = ddata[i+look_back,0]/100;
		if (i > 0):
			if (p > 0): # long a position for a day
				rtrain[i] = rtrain[i-1] * (1+r);
			elif (p <= 0): # short a position for a day
				rtrain[i] = rtrain[i-1] * (1-r);
	# end for
	rettrdata = (data[traPredict.shape[0]+look_back] - data[look_back])/data[look_back] - 1;

	###############################################################
	# for test -- 
	# at the very begin your account has 100$
	#
	for i in range(testPredict.shape[0]):
		# at day i-1, right before the end of the day, the close price of the day i-1 is known
		#	then the past 20 day history are known. 
		#	run the classifier to predict ith day stock is increase (== 1) or decrease (==0)
		#
		p = tstPredict[i];
		r = ddata[i+len(trainPredict)+(look_back*2)+1,0]/100;
		if (i > 0): # skip the first day
			if (p > 0): # the day i-1 prediction is increase --> 
						# 1. at day i-1 right before the end of the day, close all your position first
						#    close position means --if you brought stock the previous dat then sell it at closing 
						#							price; if you sold stock the previous day then buy it back at
						#							close price
						# 2. buy stock with all the money in your account
						#
						# as the result of you buy, at the end of day i, the stock change is r. The value of
						# of the account becomes day i-1 value * (1 + r) at the end of day i
				rtest[i] = rtest[i-1] * (1+r);
			elif (p <= 0): ## the day i-1 prediction is decrease --> 
						# 1. at day i-1 right before the end of the day, close all your position first
						# 2. sell stock which you do not own (call short - you need buy iot back the next day)
						#
						# as the result of you sell, at the end of day i, the stock change is r. you need buy it back.
						# the gain is -r. The value of the account becomes day i-1 value * (1 - r) at the end 
						# of day i
				rtest[i] = rtest[i-1] * (1-r);
	# end for
	# rtest contain the value of you account based on your trading

	rettsdata = data[len(data)-1, 0]/data[len(traPredict)+(look_back*2)+1,0] - 1;

	return rtrain, rtest, rettrdata, rettsdata
#end calcRetirn

def showResult ( sname, modelType, trainPredict, testPredict, trainY, testY, data, ddata, scaler, look_back=20 ):
	traPredict = np.reshape(trainPredict, len(trainPredict));
	traY =  np.reshape(trainY, len(trainY))
	tstPredict =  np.reshape(testPredict,len(testPredict))
	tstY =  np.reshape(testY, len(testY))

	# calculate root mean squared error
	if (modelType == "SVM") or (modelType == "LSTMC") or (modelType == "CNN"):
		trainScore = 1-zero_one_loss(traY, traPredict)
		print('Train accuracy: %.2f ' % (trainScore))
		testScore = 1-zero_one_loss(tstY, tstPredict)
		print('Test accuracy: %.2f ' % (testScore))
	else:
		traPredict = np.reshape(scaler.inverse_transform(np.reshape(traPredict, (len(traPredict),1))), (len(traPredict)));
		traY = np.reshape(scaler.inverse_transform(np.reshape(traY, (len(traY),1))), (len(traY)));
		tstPredict = np.reshape(scaler.inverse_transform(np.reshape(tstPredict, (len(tstPredict),1))), (len(tstPredict)));
		tstY =  np.reshape(scaler.inverse_transform(np.reshape(tstY, (len(tstY),1))), (len(tstY)));

		trainScore = math.sqrt(mean_squared_error(traY, traPredict))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(tstY, tstPredict))
		print('Test Score: %.2f RMSE' % (testScore))


		# shift train predictions for plotting
		traPredictPlot = np.zeros((len(data)))
		traPredictPlot[:] = np.nan
		traPredictPlot[look_back:len(traPredict)+look_back] = traPredict

		# shift test predictions for plotting
		tstPredictPlot = np.zeros((len(data)))
		tstPredictPlot[:] = np.nan
		tstPredictPlot[len(traPredict)+(look_back*2)+1:len(data)-1] = tstPredict

		# plot baseline and predictions
		plt.figure(num=None, figsize=(8, 6), dpi=150);
		plt.plot(ddata, 'b-' )
		plt.plot(traPredictPlot, 'g--')
		plt.plot(tstPredictPlot, 'r-')
		plt.title ( sname + '-' + modelType + ' testScore {0}'.format(testScore)  )
		plt.savefig ( sname + '-' + modelType + "-rate.jpg" )
		plt.show()

		# plot baseline and predictions
		tPlot = np.zeros((len(data)))
		tPlot[:] = np.nan
		tP = traPredict/100 + 1;
		tP = traY/100 + 1;
		tP = np.cumprod(tP)
		tP = data[look_back,0] * tP;
		tPlot[look_back:len(traPredict)+look_back] = tP
		sPlot = np.zeros((len(data)))
		sPlot[:] = np.nan
		sP = tstPredict/100 + 1;
		sP = np.cumprod(sP)
		sP = data[len(traPredict)+(look_back*2)+1] * sP;
		sPlot[len(traPredict)+(look_back*2)+1:len(data)-1] = sP

		plt.figure(num=None, figsize=(8, 6), dpi=150);
		plt.plot(data, 'b-' )
		plt.plot(tPlot, 'g--')
		plt.plot(sPlot, 'r-')
		plt.title ( sname + '-' + modelType + ' testScore {0}'.format(testScore)  )
		plt.savefig ( sname + '-' + modelType + "-price.jpeg" )
		plt.show()
	# end if modelType

	rtrain, rtest, rettrdata, rettsdata = calcReturn ( trainPredict, testPredict, data, ddata, look_back=look_back )


	plotOnlyPredict = 1; # only plot predict value against market value

	if (plotOnlyPredict):
		srPlot = np.zeros((len(rtest)))
		srPlot[:] = np.nan
		srP = rtest;
		srP = data[len(traPredict)+(look_back*2)+1,0]/100 * srP ;
		srPlot[:] = srP
		srdata = data[len(traPredict)+(look_back*2)+1:len(data)-1,:]

		# the red curve is the value of your account at each day - plotted in relative to the stock price
		# at the begin of the test period (you can thin the blue line is the value of your account you buy
		# at the beging of the test period and never do anything again.

		plt.figure(num=None, figsize=(8, 6), dpi=150);
		plt.plot(srdata, 'b-', srPlot, 'r-' )
		plt.title ( sname + '-' + modelType + ' trade return {0:.2f}% vs market return {1:.2f}%'.format(rtest[-1]-100, rettsdata*100 ) )
		plt.savefig ( sname + '-' + modelType + "-rate-price.jpeg" )
		plt.show()
	else:
		trPlot = np.zeros((len(data)))
		trPlot[:] = np.nan
		trP = rtrain;
		trP = data[look_back,0]/100 * trP;
		trPlot[look_back:len(traPredict)+look_back] = trP
		srPlot = np.zeros((len(data)))
		srPlot[:] = np.nan
		srP = rtest;
		srP = data[len(traPredict)+(look_back*2)+1,0]/100 * srP ;
		srPlot[len(traPredict)+(look_back*2)+1:len(data)-1] = srP

		plt.figure(num=None, figsize=(8, 6), dpi=150);
		plt.plot(data, 'b-' )
		plt.plot(trPlot, 'g--')
		plt.plot(srPlot, 'r-')
		plt.title ( sname + '-' + modelType + ' trade return {0:.2f}% vs market return {1:.2f}%'.format(rtest[-1]-100, rettsdata*100 ) )
		plt.savefig ( sname + '-' + modelType + "-rate-price.jpeg" )
		plt.show()
		

# endof of showResult

start="1990-01-01"
sname = 'AAPL'
#sname = '^GSPC'
look_back = 20
modelType = 'LSTM'
modelType = 'SVR'
#modelType = 'SVM'
#modelType = 'LSTMC'
#modelType = 'CNN'

trainX, trainY, testX, testY, data, ddata, scaler = create_sample(modelType, look_back=look_back, start=start)

model = createModel ( modelType, seqSize=look_back )
model = trainModel ( modelType, model, trainX, trainY )

# make predictions
trainPredict = testModel ( modelType, model, trainX )
testPredict = testModel ( modelType, model, testX)

showResult ( sname, modelType, trainPredict, testPredict, trainY, testY, data, ddata, scaler, look_back=20 )

