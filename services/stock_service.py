import yfinance as yf


def get_stock_data(symbol):

    try:
        stock = yf.Ticker(symbol)

        data = stock.history(period="10yr")

        if data.empty:
            return None

        return data

    except Exception as e:
        print("Stock data error:", e)
        return None