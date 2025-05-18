# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('weather.csv')
data.head()
data['Time'] = pd.date_range(start='2012-01-01', periods=len(data), freq='M')
data.set_index('date', inplace=True)
time_series = data['wind']
plt.figure(figsize=(10, 6))
plt.plot(time_series)
plt.title('Wind Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Wind Level')
plt.show()
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

test_stationarity(time_series)
time_series_diff = time_series.diff().dropna()
print("\nAfter Differencing:")
test_stationarity(time_series_diff)
# Set initial SARIMA parameters (p, d, q, P, D, Q, m)
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12  # m = 12 for monthly seasonality if applicable
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)
print(sarima_fit.summary())
forecast_steps = 12  # Number of periods to forecast
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Ensure time series index is datetime and timezone-naive
time_series.index = pd.to_datetime(time_series.index).tz_localize(None)
forecast.predicted_mean.index = pd.to_datetime(forecast.predicted_mean.index).tz_localize(None)
forecast_ci.index = pd.to_datetime(forecast_ci.index).tz_localize(None)


plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of Wind Levels')
plt.xlabel('Date')
plt.ylabel('Wind Level')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
test_data = time_series[-forecast_steps:]
pred_data = forecast.predicted_mean[:len(test_data)]
mae = mean_absolute_error(test_data, pred_data)
print('Mean Absolute Error:', mae)
```

### OUTPUT:

#### Time Series Plot:

![download](https://github.com/user-attachments/assets/b5d934e9-3e4b-400a-9948-0480d25814de)

#### After Differencing:

![image](https://github.com/user-attachments/assets/b3d6e4f5-336d-4aea-b990-90539a5a0ed2)

#### SARIMA Forecast:

![download](https://github.com/user-attachments/assets/3b4fc5f4-4934-4f04-b4c4-a5d015e114a3)

#### Mean Absolute Error:

![image](https://github.com/user-attachments/assets/215ce5e4-1382-422d-b5b5-cde4aeff3a3e)

### RESULT:
Thus the program run successfully based on the SARIMA model.
