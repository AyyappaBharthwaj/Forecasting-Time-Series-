##################################### problem1 ##################################################
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

dataset = pd.read_excel("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/Airlines Data.xlsx")
dataset.columns
#Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'],infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])

from datetime import datetime
indexedDataset.head(5)

#plot graph
plt.xlabel("Date")
plt.ylabel("Number of air passengers")
plt.plot(indexedDataset)

# 2 methods used To check the stationarity of the data Rolling Statistics and Dickey-Fuller
#Determining Rolling Statistics
rolmean = indexedDataset.rolling(window=12).mean()

rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

#plot Rolling Statistics:
orig = plt.plot(indexedDataset,color = 'blue',label = 'Original')
mean = plt.plot(rolmean,color = 'red',label = 'Rolling Mean')   
std = plt.plot(rolstd,color = 'black',label = 'Rolling Std')   
plt.legend(loc= 'best')
plt.title('Rolling Mean & Standard Deviation')

#Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller    

print("Result of Dickey-Fuller test:")
dftest = adfuller(indexedDataset['Passengers'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=["Test Statitics","p-value","#Lags Used","Number of Observations Used"])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)    

#Estimated trend
indexedDataset_logscale = np.log(indexedDataset)
plt.plot(indexedDataset_logscale)

movingAverage = indexedDataset_logscale.rolling(window = 12).mean()
movingSTD= indexedDataset_logscale.rolling(window = 12).std()
plt.plot(indexedDataset_logscale)
plt.plot(movingAverage,color = "red")

datasetLogScaleMinusMovingAverage = indexedDataset_logscale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove Nan values
datasetLogScaleMinusMovingAverage.dropna(inplace = True)
datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import adfuller   
def test_stationarity(timeseries):
    
    #determining rolling statistics
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD= timeseries.rolling(window = 12).std()
    
    #plot rolling statistics:
    orig = plt.plot(timeseries,color = 'blue',label = 'Original')
    mean = plt.plot(movingAverage,color = 'red',label = 'Rolling Mean')   
    std = plt.plot(movingSTD,color = 'black',label = 'Rolling Std')   
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
        
    #Perform Dickey-Fuller test:
    print("Result of Dickey-Fuller test:")
    dftest = adfuller(timeseries['Passengers'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=["Test Statitics","p-value","#Lags Used","Number of Observations Used"])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)    

test_stationarity(datasetLogScaleMinusMovingAverage)   

exponentialDecayWeightedAverage = indexedDataset_logscale.ewm(halflife=12, min_periods = 0,adjust = True).mean() 
plt.plot(indexedDataset_logscale)
plt.plot(exponentialDecayWeightedAverage, color ="red")

datasetLogScaleMinusexponentialDecayWeightedAverage = indexedDataset_logscale - exponentialDecayWeightedAverage 
test_stationarity(datasetLogScaleMinusexponentialDecayWeightedAverage)

datasetLogDiffShifting = indexedDataset_logscale - indexedDataset_logscale.shift() 
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace = True)
test_stationarity(datasetLogDiffShifting)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logscale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logscale, label = "Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend,label = "Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal,label="Seasonality")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual,label="Residuals")
plt.legend(loc="best")
plt.tight_layout()

#decomposedLogData = residual
#decomposedLogData.dropna(inplace=True)
#test_stationarity(decomposedLogData)

#ACFand PACF plots: In order to find q & p value
from statsmodels.tsa.stattools import acf,pacf

lag_acf = acf(datasetLogDiffShifting, nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting, nlags = 20, method = 'ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title("Autocorrelation Function")

#plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title("Partial Autocorrelation Function")
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_AR.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

#MA MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,0))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_MA.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

#Combining AR and MA model
#MA MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_ARIMA.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(prediction_ARIMA_diff.head())

#convert to cumulative sum
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum.head())

prediction_ARIMA_log = pd.Series(indexedDataset_logscale["Passengers"].iloc[0], index=indexedDataset_logscale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()

prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)

indexedDataset_logscale

results_ARIMA.plot_predict(1,156)
results_ARIMA.forecast(steps=60)

########################################## problem2 ############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

cocacola = pd.read_excel("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/CocaCola_Sales_Rawdata.xlsx")

cocacola.Sales.plot() # time series plot 

# Centering moving average for the time series
cocacola.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
# tsa_plots.plot_pacf(cocacola.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = cocacola.head(38)
Test = cocacola.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/new_cocacola_data.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

###################################### problem3 ##########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

plastic_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/PlasticSales.csv")

plastic_data.Sales.plot() # time series plot 

# Centering moving average for the time series
plastic_data.Sales.plot(label = "org")
for i in range(2, 9, 2):
    plastic_data["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic_data.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(plastic_data.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(plastic_data.Sales, lags = 4)
# tsa_plots.plot_pacf(plastic_data.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = plastic_data.head(49)
Test = plastic_data.tail(60)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(plastic_data["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/New_plastic_sales_data.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

############################################## problem4 ###############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

solarpower_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/solarpower_cumuldaybyday2.csv")
solarpower_data.columns

solarpower_data.cum_power.plot() # time series plot 

# Centering moving average for the time series
solarpower_data.cum_power.plot(label = "org")
for i in range(2, 9, 2):
    solarpower_data["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(solarpower_data.cum_power, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(solarpower_data.cum_power, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(solarpower_data.cum_power, lags = 4)
# tsa_plots.plot_pacf(plastic_data.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = solarpower_data.head(2047)
Test = solarpower_data.tail(2558)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.cum_power) 

# Holt method 
hw_model = Holt(Train["cum_power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.cum_power) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.cum_power) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.cum_power) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(solarpower_data["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/solarpower_cumuldaybyday2.csv")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

############################################ END #################################################

