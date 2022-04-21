# Stock_Projection_with_Machine_Learning

Website link: https://jordanjaner.github.io/stock-forecaster/

<img src="/templates/images/stock_main.jpg" alt="Stock Projection with Machine Learning and LSTM"/>

## Project Purpose

The goal of this project is to use previous stock stats from datasets to predict future highs and lows with machine learning. In doing so we will be able to answer questions such as what are the strongest stocks to invest in for increased gains, or which stocks to not invest in to avoid losses based on the previous stock history. We will provide an analysis and visualization to help the reader better understand the prediction of stock flow.

<img src="/templates/images/stock_project_intro.png" alt="Stock Project Intro"/>

### Research Questions
* Are the major stocks the best to invest in? 
* What are the best stocks to buy for next year? 
* What are the safest types of stocks to buy based on past and projected stats? 
* What period of the year showed the most potential to invest?

### Data Sources
For this project we used Yahoo Finance API to extract S&P 500 data. Initially we were going to train S&P 500, DOW, and NASDAQ, but we ran into problems dealing with the size of the data. Ultimately we only used S&P 500 and only used 1 year worth of data to train. We exported into a CSV and then used Google Colab for Machine Learning.
https://finance.yahoo.com/
https://datahub.io/core/nasdaq-listings
https://thecleverprogrammer.com/2020/08/22/real-time-stock-price-with-python/

### Importing Data and Preprocessing
First we code for importing the data from the CSV and beigin preprocessing.

### Import

<img src="/templates/images/import_csv.png" alt="Import Data"/>

### Preprocess

<img src="/templates/images/preprocessing.png" alt="Preprocessing"/>

### Reshaping
Next we reshape the model to setup for training.

<img src="/templates/images/reshape.png" alt="reshape"/>

### Build our LSTM Model
We then build our LSTM model to make predictions based on the data we provide from the stock.

<img src="/templates/images/build_model.png" alt="build model"/>

### Predicting
After building the model, then we can make predictions from the data.

<img src="/templates/images/predicting.png" alt="predicting"/>

Using Apple stock as our tester with 60 TimeSteps and 100 TimeSteps

<img src="/templates/images/60apple.png" alt="60 TimeSteps"/>

<img src="/templates/images/100apple.png" alt="100 TimeSteps"/>

### Conclusion and Limitations
After the creation, reshaping, and training of the model we have concluded that our model is very accurate. For improved predictions, we have trained this model on stock price data for companies in the same sector, region, subsidiaries, etc. (in our analysis big techs). Therefore, model prediction results on stock prices of companies out of this sector may not be quite accurate. We also could have included more tech companies in our training, and make our model sector specific for better prediction results. Our Machine learning model only asks user to input their stock of interest. We could also ask user to input the time period they were interested to look at. However, for the interest of time we set the period as a constant and not a user input variable.
