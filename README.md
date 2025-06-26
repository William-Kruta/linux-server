# Cli Help

### Candles

python cli.py download_candles AAPL --interval 1m
python cli.py download_candles_using_list watchlist --interval 1m

### Statements

python cli.py download_statements AAPL --period Q
python cli.py download_statements_using_list watchlist --period A

### Options

python cli.py download_options AAPL
python cli.py download_options_using_list watchlist

--- 

# Server Set Up 

Execute `python server.py` in the command line to run the server. 
You can configure the port the server runs on, by default `8000`.

---

# API Set Up
Within `linux-server/modules/api/api.py` you can find the api script. 
This script can be used on your main computer to interact with the server computer. 


Below is an example usage: 

```
api = API(server_url, time_zone="PST")

data1d = api.get_candles("AAPL", "1d") # Daily candles for AAPL
data1m = api.get_candles("AAPL", "1m") # 1-minute candles for AAPL


options = api.get_options("AAPL", all_expirations=True) # Historical Options Chain 


quarterly_income_statement = api.get_income_statement("AAPL", quarter=True) # Quarterly income statement data.
annual_income_statement = api.get_income_statement("AAPL", quarter=False) # Annual income statement data.

```


