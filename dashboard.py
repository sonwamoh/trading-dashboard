import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pandas_datareader.data import DataReader
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from stocknews import StockNews


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(" Welcome to My Trading Dashboard!")

option = st.sidebar.selectbox("Which Dashboard?", {'Analyze Stock Performance', 'Predict Future Stock Prices', 'Stock News'})
                
if option == 'Analyze Stock Performance':
    st.write("### Stock Data Visualization")
    user_input = st.text_input("Enter 4 company name and stock symbol (comma-separated tuples):", "(Apple,AAPL),(Google,GOOG),(Microsoft,MSFT),(Meta, META)")

    company_data = [tuple(pair.strip("()").split(',')) for pair in user_input.split('),(')]
    
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    start_date = st.date_input("Start Date", start)
    end_date = st.date_input("End Date", end)

    stock_data = {}
    for company_name, stock_symbol in company_data:
        stock_data[company_name] = yf.download(stock_symbol, start_date, end_date)
    
    company_list = []
    for company_name, df in stock_data.items():
        df["company_name"] = company_name
        company_list.append(df)

    df = pd.concat(company_list, axis=0)

    st.write("### Stock Data")
    st.write(df)
    
    tech_list = [i[0] for i in company_data]
    st.write("### Adjusted Closing Price")
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")
    plt.tight_layout()
    st.pyplot(plt.show())
    
    st.write("### Volume of Stock")
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {tech_list[i - 1]}")
    plt.tight_layout()
    st.pyplot(plt.show())
    
    st.write("### Daily Returns")
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

    plt.figure(figsize=(12, 7))
    
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Daily Return'].hist(bins=50)
        plt.ylabel('Daily Return')
        plt.title(f'{company_name[i - 1]}')
    st.pyplot(plt.gcf())
    
    st.write("### Moving Averages")
    ma_day = [10, 20, 50]

    for company in company_list:
        for ma in ma_day:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()

    fig, axes = plt.subplots(nrows=len(company_list), ncols=1, figsize=(15, 4 * len(company_list)))
    for i, stock in enumerate(company_list):
        ax = axes[i]
        stock[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
        ax.set_title(f"{stock['company_name'].iloc[0]}")
    fig.tight_layout()
    st.pyplot(fig)
    
    st.write("### Correlation Between Different Stock Closing Prices")
    res_df = df[['Adj Close','company_name']]
    res_df = res_df.sort_values(by=['company_name', 'Date'])
    res_df_pivoted = res_df.pivot(columns='company_name', values='Adj Close').reset_index()
    res_df_pivoted = res_df_pivoted.set_index('Date')
    tech_rets = res_df_pivoted.pct_change()
    st.pyplot(sns.pairplot(tech_rets, kind='reg'))
    
    st.write("### Correlation Between Different Stock Returns")
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock return')

    plt.subplot(2, 2, 2)
    sns.heatmap(res_df_pivoted.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock closing price') 
    st.pyplot(plt.show())
    
    st.write("### Stock Investment Risks")
    rets = tech_rets.dropna()

    area = np.pi * 20

    plt.figure(figsize=(10, 7))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(
            label, xy=(x, y), xytext=(50, 50), textcoords='offset points',
            ha='right', va='bottom',
            arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3')
        )
    st.pyplot(plt.show())

if option == 'Predict Future Stock Prices':
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    st.write("### Predict Future Stock Prices")

    select_stocks = st.text_input('Enter valid stock symbol:', 'NFLX')

    n_years = st.slider("Years of Prediction:", 1, 4)
    period = n_years * 365


    @st.cache_data
    def load_date(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data...")
    data = load_date(select_stocks)
    data_load_state.text("Loading data...done!")
        
    st.write("### Raw Data")
    st.write(data.tail())


    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'],
                    y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'],
                    y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data",
                        xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_raw_data()

    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.write("### Forecast Data")
    st.write(forecast.tail())

    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

if option == 'Stock News':
    ticker = st.text_input('Enter valid stock symbol:', 'GOOG')
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i] 
        st.write(f'Title Sentiment {title_sentiment}') 
        news_sentiment = df_news['sentiment_summary'][i] 
        st.write(f'News Sentiment {news_sentiment}')




    




    

    


    
    
    

    

    



        
    

  




    
        

    




   
    
  


    
        

    

    


    
    

