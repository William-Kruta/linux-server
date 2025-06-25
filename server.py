import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import datetime

from modules.data.candles import get_candles


app = FastAPI()

class Candle(BaseModel):
    time: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

# pretend this is where your candles live
DATA_STORE: Dict[str, List[Candle]] = {
    "BTCUSD": [
        Candle(time=datetime.datetime.utcnow(), open=60000, high=60500, low=59800, close=60200, volume=1.23),
        # … more candles …
    ]
}

@app.get("/get_candles/{symbol}/{interval}", response_model=List[Candle])
def server_get_candles(symbol: str, interval: str):
    """
    
    Args:
        symbol(str): Ticker symbol of the stock. 
        interval(str): Iterval of data. 1d is daily candles, "1m" is minute level candles. 

    Returns: 
        Dataframe
    
    """
    symbol = symbol.upper()
    data = get_candles(ticker=symbol, interval=interval)
    return data



if __name__ == "__main__":
    # listen on all interfaces so your main PC can reach it
    
    uvicorn.run(app, host="0.0.0.0", port=8000)