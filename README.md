# Genetic Algorithms Optimize Asset Allocation
# Introduction

Asset Allocation and Portfolio have been very popular topics for many years. Investors can obtain excess returns and diversify their investments by investing in different financial products.
risk. However, how to select high-return, low-risk investment portfolios from the market is really difficult.
knowledge.

As we all know, when investing in financial products, high returns come with high risks. Nobel Prize Winner in Economics
Markowitz proposed the concept of investment portfolio in 1952, using the metaphor of "eggs should not be put in the same basket" to illustrate the importance of risk diversification; using diversification to reduce the risk of investing in commodities, it is expected to achieve the lowest risk effect. . Among them, expected return is often defined as the simple average of daily profits and losses in the past period; risk, also known as volatility, is often defined as the standard deviation of daily profits and losses in the past period. The larger the standard deviation, the greater the standard deviation of the asset in the past. The greater the fluctuation, the greater the risk.

By allocating assets with low correlations in a portfolio, you can effectively diversify volatility. During significant downside risks, unlike single assets, other positions in the portfolio may have bullish trends, thereby hedging the losses. This is equivalent to installing shock absorbers for your investment. A portfolio with relatively small losses will tend to have a stable profit and loss curve, which is beneficial for investors to use leverage.

Since the portfolio is a weighted result, its cumulative return will lie between the upper and lower bounds of the cumulative returns of the individual assets within the portfolio. However, with effective risk diversification, the volatility from leverage will be at a similar level to investing in a single asset without risk diversification, while still enjoying multiple times the excess returns.

Whether obtaining higher returns with the same risk or significantly reducing risk with the same returns, portfolio management is greatly favored by investors. However, creating an optimal portfolio poses a significant challenge for average investors with limited computational power and financial knowledge—they cannot guarantee the quality of their investment portfolios. Therefore, this research aims to solve this issue through the use of genetic algorithms.

# Related Research

In many quantitative trading hedge funds and trading rooms, mathematical statistics and programming are widely used to solve trading problems. Asset allocation is an important area of research within this scope.

Although this study did not reference other papers during its execution, we examined related topics after completing the research. We found that similar algorithms or machine learning attempts have been used in asset allocation in academic studies. For example, Fan-Chun Meng (2019) utilized DDPG and PPO deep reinforcement learning algorithms, Shang-Ren Lin (2020) used PPO deep reinforcement learning algorithms, and Zihao Zhang, Stefan Zohren, and Stephen Roberts (2020 FALL) employed deep learning in the Journal of Financial Data Science.

Regarding research outcomes, we discovered that while the deep learning methods used by Zihao Zhang, Stefan Zohren, and Stephen Roberts yielded good results in their papers, the backtesting instruments used, such as the non-tradable VIX index (fear index), have limitations. We believe that the accuracy and stability of such backtesting results for non-tradable market instruments require further scrutiny.

Therefore, before commencing this study, we decided to seek historical data of tradable instruments for backtesting. Regardless of the research outcomes, this approach ensures that our research methodology has at least a practical basis for evaluation.

# Methodology

__(1) Analytical Methods__

Genetic Algorithm (GA) draws inspiration from evolutionary biology phenomena such as inheritance, selection, mutation, and crossover. GA excels at solving global optimization problems and is often used to tackle practical engineering issues like logistics management, production scheduling, and computer-aided design. Compared to traditional optimization algorithms, one of GA's strengths is its ability to escape local optima and find global optima. Moreover, GA allows the use of highly complex fitness functions (or objective functions) and can impose restrictions on variable ranges. In contrast, traditional optimization algorithms may encounter significantly more complex processes when dealing with variable range constraints.


Due to its high flexibility and wide applicability, we aim to apply GA's characteristics to the financial domain to construct an optimized asset portfolio.

__(2) Performance Indicators__

To select the optimal investment portfolio, it is essential first to define the performance metrics. This study prioritizes the "Sharpe Ratio" as the primary indicator, optimizing for it as the evolutionary direction. Simultaneously, "Maximum Drawdown," "Average Drawdown," and "Risk-Reward Ratio" will also be used as references to measure the quality of the investment portfolio.

__What is the Sharpe Ratio?__

When investors choose investment targets, they typically first observe the historical cumulative returns of the asset. However, high-return assets with significant volatility, or those that oscillate for most of the investment period and only increase in the last year, like Taiwan shipping stocks post-pandemic, do not make for good investment portfolios. Investors might face substantial losses or incur time costs without seeing results, leading to a loss of confidence and withdrawal of funds, missing out on potential gains. Therefore, for an investment portfolio, seeking low risk volatility with stable growth is a better indicator than mere cumulative returns. Academics and practitioners often use the "Sharpe Ratio" to measure the risk and return of an asset or investment portfolio. The formula for Sharpe Ratio is: $$Sharpe Ratio = \frac{(Expected Return - Risk-free Interest)}{Risk Rate}$$ This is defined as the expected value of the investment return relative to the risk-free return, divided by the standard deviation (volatility) of the investment, indicating the extra return per unit of risk taken.

__Drawdown__

Besides the Sharpe Ratio, Drawdown is also a commonly used metric in practical investment trading. It is defined as the decline from the highest point of the return curve at any given time, essentially the loss relative to the highest value that could have been realized but wasn't. For example, if a stock is bought at $50 and reaches a new high of $100 by the previous day's close but is not sold, and drops to $98 by today's close, the $2 loss is the drawdown. Maximum Drawdown, Average Drawdown, and the Risk-Reward Ratio (cumulative return divided by the maximum drawdown) are frequently used performance metrics in practical trading. A large drawdown can cause traders to lose confidence and withdraw their positions, representing another form of risk.

__(3) Analyze Goals__

Our research focuses on multiple assets with weight adjustments at regular intervals, such as quarterly, annually, or other specified periods. The goal is to create a portfolio with the best Sharpe ratio and compare it to benchmarks (average allocation, random allocation). This study aims to maximize the Sharpe ratio, and ultimately, the selected portfolios of each team member will be presented.
The data for the entire period is sourced from Yahoo Finance, covering the period from January 1, 2005, to January 1, 2022. The assets are as follows (Yahoo Finance ticker symbols in parentheses):

Cocoa Futures (CC=F)<br>
NASDAQ Futures (NQ=F)<br>
0050 ETF (0050.TW)<br>
TSMC Stock (2330.TW)<br>
Gold Futures (GC=F)<br>
U.S. Treasury Futures (ZB=F)<br>

__Crawler Function Implementation Code__

The YahooData function is used to fetch the closing price data for a given date range and stock or futures ticker symbol. The Generate_Data function takes a list of stock ticker symbols as input and consolidates all closing prices into a single DataFrame.

```python
def YahooData(ticker, start, end):
  headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
  }

  url = "https://query1.finance.yahoo.com/v7/finance/download/" + str(ticker)
  x = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
  y = int(datetime.strptime(end, '%Y-%m-%d').timestamp())
  url += "?period1=" + str(x) + "&period2=" + str(y) + "&interval=1d&events=history&includeAdjustedClose=true"

  r = requests.get(url, headers=headers)
  pad = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
  return pad['Close'].pct_change()


def generate_data(t_list, start_day, end_day):
  print('start loading')
  df = pd.DataFrame()
  for ticker in t_list:
    temp_stock = YahooData(ticker, start_day, end_day)
    print(ticker, 'loaded')
    df = pd.concat([df, temp_stock], axis=1)
  print('loading finish')
  df.columns = t_list
  df = df.fillna(0)
  return df
```

__(4)Analysis Architecture__

We split the entire dataset into several training and testing datasets, updating them at fixed intervals. (The analysis method will be explained in the implementation process later.) The weights trained on the training data are then applied to the corresponding testing data to observe the portfolio's performance. This approach achieves practical trading effects, avoiding overfitting or foresight issues.

![My Image](影像.jpeg)

__(5)Algorithm: Genetic Algorithm__

![My Image](pic2.JPEG)

