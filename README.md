# Bitcoin-Price-Prediction

It has been noticed that there is a rise in the popularity of cryptocurrencies
among institutional and retail users alike and their market capitalization has
increased dramatically. In terms of market capitalization, Bitcoin is the most
traded and biggest cryptocurrency. Retail investors are far more important
in Bitcoin trading than institutional investors are in traditional asset trading
as was seen in a research (Nathan et al. (2021)[1]). 

Bitcoin is among one of
the digital assets that are not reliant on material needs like coal and iron ore
and because of this, the price of Bitcoin is more vulnerable to changes in the
market sentiment.To give an example, the Bitcoin price went up by around
5% on March 2021, and one of the reason was that Elon Musk announced
that Tesla will accept Bitcoin as payment. It also fell around 9% on May
2021, because there was a tweet from Elon Musk regarding the consumption
of energy during the process of Bitcoin mining.

In our project, we implemented a multimodal model to predict the fluctuations of Bitcoin price which was suggested by an existing research. In this
case, data from Twitter as well as large database of price data including
technical indicator and linked asset price were also used. Instances of studies
are there to find out relationship of sentiments with Bitcoin values. But doing so may lead to missing lot of potentially helpful data.

Therefore, in this
study, we implemented a strategy to incorporate the complete tweet content
into a Bidirectional Encoder Representations from Transformers model and
feed it into the predictive model. The model was seen to improve further by
using historical candlestick (OHLCV) data and technical indicators, as well
as associated asset values such as Ethereum and Gold.
