# This project is about analyzing stock price of a particular company and for this particular project, we are using the Amazon’s stock prices from January 2015 till September 2018. We gathered this data by using the technique know as web scrapping.
# For scrapping of the data from the web, we used a package called “quantmod”. This package basically provides a framework for quantitative financial modeling and trading. This particular package is designed specifically for traders in order for them to do quantitative analysis so as to help them in decision making. It can help traders to analyze when to sell a stock and when to buy a stock so is to achieve maximum profit possible, regarding which we talked in the presentation in class, also we will be covering in the further part of this report.

install.packages("quantmod")
library(quantmod)
library(forecast)

# The function “getSymbols()” of the “quantmod” package helps us to scarp the data from the web. The function is provided with the tick word of a particular company (AMZN for amazon in this case) and the function returns data of the stock of that particular company. The function scraps this data from online sources like yahoo and google.
# The stock market data that this function returns is in xts format which is generally efficient to deal with but however, we have converted it into data frame later on and performed analysis on it.
amz <- getSymbols("AMZN", auto.assign=F,
                  from = "2015-01-01", to = "2018-10-19")
#Using function head, class, summary and str to get clear idea about the data and be familiar with all the variable.
# Our amazon dataset has 6 columns which are AMZN.volume(stocks traded on a day), AMZN.open (opening price of the stock) AMZN.high (highest price of the stock on a day) AMZN.low (lowest price of the stock on a day) AMZN.close (closing price of the stock on a day) and AMZN.adjusted. Each row in the dataset contains a trading day.

head(amz)
class(amz)
summary(amz)
str(amz)

#plotting the graph for Amazon data which is in xts format just to have a look at data and be familiar with.
chartSeries(amz, type = "line")

#Our dataset contains 5 time series (low, high, open, close and volume), for the sake of this project we will be analyzing the data on AMZN.open time series.
amz1 <- as.data.frame(amz)
amz_ts <- ts(amz1$AMZN.Open, start = c(2015,1), end = c(2018,10), frequency = 12)
class(amz_ts)
plot(amz_ts, main = 'Timeseries', ylab = "Amazon_Open")

#When we decomposed the series into seasonal, trend and remainder we found that our data has an increasing trend over the years. We also found that our data has a seasonal component to it which we can depict from the graph that, every year in the months of 5-6 we see a dip in the price of stock and then again a sudden increase in the stock price in the reoccurring month.
stl1 <- stl(amz_ts, s.window = 12)
plot(stl1, col = "red ", main="Seasonal Decomposition")

#Since our data has a seasonal component, in order to deal with such series, we need to stabilize (stationary) the series. And hence we use the diff() function to stabilize the series.
deseasonal_cnt <- seasadj(stl1)
count_d1 = diff(deseasonal_cnt, differences = 1)
plot(count_d1)

#Here, we plotted the acf and pacf which is auto correlation and partial auto correlation which helps us to determine whether the data is stationary or not and it also helps us to determine to predict the parameter for ARIMA model, firstly we plotted with the non stationary time series data which is our actual data, where we found that there was no an oscillating pattern in ACF and where not able to determine any thing from it.
ACF1 <- acf(amz_ts, col="blue", main="ACF Decomposition")
Pacf1 <- pacf(amz_ts, col="blue", main= "PACF Decomposition")

#So used the differencing 1 on the data which helps to rejects the null hypotheses of non-stationarity data. Plotting ACF and PACF using differenced series, we see an oscillating pattern. So it tells us differencing of order 1 terms is needed and should be included in model also.
ACF <- acf(count_d1, col="blue", main="ACF Decomposition")
Pacf <- pacf(count_d1, col="blue", main= "PACF Decomposition")

#Converting the time series into training and test time series data to predict the result and get accuracy of the model.
# Train test split for holt-winters.
train_ts <- window(amz_ts, start = c(2015,1), end = c(2017,10))
test_ts <- window(amz_ts, start = c(2017,11))

# Train test split for ARIMA model.
train_ts1 <- window(amz_ts - stl1$time.series[,1], start = c(2015,1), end = c(2017,10))
print(train_ts1)
test_ts1 <- window(amz_ts - stl1$time.series[,1], start = c(2017,11))
print(test_ts1)

#Here, we have time series with increasing trend and no seasonality, so Holt winter exponential smoothing helps to make short-term forecasts with accurate results. Holt winter exponential smoothing estimates the level at the current time point and slope at the current time point. The parameters of alpha and beta have values between 0 and 1. So, here Smoothing is controlled by two parameters, alpha which will help to estimate the level at the current time point, and beta which helps for estimating the slope b of the trend component at the current time point. 
#Here implementing the holt winter we got to know alpha and beta values were high with 0.9 for alpha and 1 for beta. So we can say that high weight is placed at the most recent observations when making forecasts of future values. 
# Using the holt winter model
fit = HoltWinters(train_ts, alpha = 0.9)
forecast <- forecast(fit, 12)
plot(forecast, main = "Holt-Winter", ylab="Amazon_Open")
lines(test_ts, col = "red")

#Here we tried with simple auto arima by not stating seasonal parameter it showed us straight line at forecasting data. As the model is assuming a series with no seasonality, and is differencing the original non-stationary data. So, to improve the model, we added seasonal component and force to take differencing values 1 to auto arima model. Re-fitting the model on the same data, we see that forecasting and with less AIC compared to earlier.
# Using auto arima model
fit_auto <- auto.arima(train_ts1,seasonal=T, d= 1, D=1)
fit_auto
forecasted_ts1 <- forecast(fit_auto, 12)
print(forecasted_ts1$mean)
plot(forecasted_ts1, main = "Auto-Arima", ylab="Amazon_Open")
lines(test_ts1, col = "red")

# After that we decided to move forward with simple arima by fitting the parameter by over self. We determined auto regression and moving average by looking at the lags of the ACF and PACF. By taking small p 1 (AR) from PACF and q 0 (MA) from ACF and putting d = 1 for differencing the time series. And same making a trail and error method for P, D, Q and checking the lowest AIC we found for 1,1,0 and 0,1,0 i.e. 179. 
# Getting the idea of selecting parameters from Oracle data Science website 
# Using Arima model
amzarima2 = Arima(train_ts1, order = c(1,1,0), seasonal = list(order=c(0,1,0), period = 12))
amzarima2
forecasted_arima <- forecast(amzarima2, 12)
plot(forecasted_arima, main = "Arima")
lines(test_ts1, col = "red")

#	We will compare all the 3 methods by calculating the mean absolute error to check the accuracy.
# Accuracy for Arima
arima_accuracy <- 1/12 *sum(abs(forecasted_arima$mean - test_ts1))
print(arima_accuracy)

# Accuracy for auto.arima
auto.arima_accuracy <- 1/12 *sum(abs(forecasted_ts1$mean - test_ts1))
print(auto.arima_accuracy)

# Accuracy for Holt-Winters
HoltWinters_accuracy <- 1/12 *sum(abs(forecast$mean - test_ts))
print(HoltWinters_accuracy)

#So after implementing the accuracy we found that Arima model has the least mean absolute error so arima is the most accurate of all the three method that we have used for evaluating the best fit. Then the auto arima and least accurate from the 3 is holtwinters.
# It is also evident from the autoplot which displays the forecast of all the methods and in that arima is the best fit, then auto arima and last holt winters.
# The auto plot has the comparison of the actual value with the value created from the model.
# In this the auto layer from the forecast package will be used along with the three fitted models

# In order to solve analytical question about the pattern within the trading days we need to convert the irregular spaced data to regular spaced data so that we can use the frequency parameter for finding the pattern
# First step is to use the data frame amz1 that we have already created
# In that data frame date is just available as a row name, so we create a new column named date, and convert the new column date into date format
# Adding the rownames as date
amz1$Date = rownames(amz1)
amz1$Date
amz1$Date = as.Date(amz1$Date)
head(amz1)

#Next step is to create object my_dates which will contain all the dates for a specified period, we will use seq.date function by passing the parameters “from” is the start date and “to” is the end date and “by” is the step size.
#Now we will convert the object my_date object into data frame
#And lastly, we will merge the original data frame amz1 which contains the data for the traded days and the new my_date dataframe consist all the dates within the period, and the new data frame my_data created contains all the trading and the non-trading days with NA.
my_dates = seq.Date(from = as.Date("2015-01-02"), 
                   to = as.Date("2018-10-19"), 
                   by = 1)
my_dates = data.frame(Date = my_dates)
my_data = merge(amz1, my_dates, by = "Date", all.y = T)
head(my_data)

#For getting the pattern for a trading week, we need a regularly spaced dataset with a frequency of 5, because there are 5 trading days a week, 
#First step is to chop some of the first few observations in order to a get the dataset started from a full trading week and for that we do index operation.
#Now the data set starts from January 5th 2015 which is Monday.
#Now as we have a regularly spaced data 
#Now we will get rid of the Sunday and Saturday by using the seq function, and subtract the sequence from the data, in seq function we will pass the parameter “from” indicate the first observation to be removed, “to” specify the last observation of the dataset and “by” is the step size which will be 7 ,as 7 is for Sunday in the weekly cycle. and same process for removing Saturday by changing the parameter to 6. Now there are some missing values remaining which are the holidays, and for that we will perform imputation by using last observation carried forward (locf). Now the data is regularly spaced data with no NA values with a frequency of 5. Which can be used to find pattern between the trading week.

# Removing initial days to start on monday
my_data = my_data[4:1387,]
# Removing sundays,
my_data = my_data[-(seq(from = 7, to = nrow(my_data), by = 7)),]
# Removing saturdays,
my_data = my_data[-(seq(from = 6, to = nrow(my_data), by = 6)),]
# Using last observatoin carried forward imputation
my_data = na.locf(my_data)
head(my_data)

# Now that the data is regular, we will find pattern between the trading days of a week with high value of the stock price for each day. We will call the timeseries for the highest price and convert it to numeric and using frequency 5 as we have a full trading week.
# For visualization we will use month plot, which is the line chart for all the highest price for each of 5 days of the week

# Putting the Highprice into a weekly time series
highest_price = ts(as.numeric(my_data$AMZN.High), 
                  frequency = 5)
median(highest_price)
# seasonplot(highestprice, season.labels = c("Mon", "Tue", "Wed", "Thu", "Fri"))
monthplot(highest_price, base = median, col.base = "red")

# Now we will plot a month plot for the lowest price and compare it with the plot for the highest price.
# Comparison with the low prices
par(mfrow = c(1,2))
lowest_price = ts(as.numeric(my_data$AMZN.Close), 
                 frequency = 5)
median(lowest_price)
monthplot(lowest_price, base = median, col.base = "red")

monthplot(highest_price, base = median, col.base = "red")
par(mfrow = c(1,1))

#We can see a minute difference between the days of a week. From the month plot for lowest price we can conclude that it is better to buy the stocks on Tuesday, Wednesday and Thursday as the probability of the median price of the stock is lowest on these three days. From the month plot for high price we can conclude that is better to sell stocks on Monday and Friday as there is high probability of getting a higher price for the stock.

# References:
# Rob J Hyndman & George Athanasopoulos  “FORECASTING: PRINCIPLES AND PRACTICE 2nd edition, May 2018” Retrieved from https://otexts.org/fpp2/accuracy.html  
# Ruslana Dalinina (January 10, 2017) “Introduction to Forecasting with ARIMA in R” Retrieved from https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials  
# Jason Brownlee (February 6, 2017) “A Gentle Introduction to Autocorrelation and Partial Autocorrelation” Retrieved from https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/ 
# Retrieved from https://stats.stackexchange.com/questions/355538/why-does-minimizing-the-mae-lead-to-forecasting-the-median-and-not-the-mean 
# Ruslana Dalinina (January 10, 2017) “Introduction to Forecasting with ARIMA in R” Retrieved from https://www.datascience.com/blog/introduction-to-forecasting-with-arima-in-r-learn-data-science-tutorials

