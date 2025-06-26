import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import datetime

from modules.data.candles import get_candles
from modules.data.options import get_options_chain
from modules.data.news import get_news
from modules.data.financial_statements import get_income_statement, get_balance_sheet, get_cash_flow
from config.config import get_candles_path


DB_PATH = get_candles_path()


app = FastAPI()

class Candle(BaseModel):
    time: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@app.get("/get_candles/{symbol}", response_model=List)
def server_get_candles(symbol: str, interval: str):
    """
    
    Args:
        symbol(str): Ticker symbol of the stock. 
        interval(str): Iterval of data. 1d is daily candles, "1m" is minute level candles. 

    Returns: 
        Dataframe
    
    """
    symbol = symbol.upper()
    data = get_candles(ticker=symbol, interval=interval, db_path=DB_PATH)
    data = data.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)

@app.get("/get_options/{symbol}", response_model=List)
def server_get_options(symbol: str, all_expirations: bool = True): 
    chain = get_options_chain(symbol.upper(), DB_PATH, all_expirations)
    data = chain.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)

@app.get("/get_income_statement/{symbol}")
def server_get_income_statement(symbol: str, quarter: bool = False): 
    data = get_income_statement(symbol.upper(), quarter, DB_PATH)
    data = data.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)

@app.get("/get_balance_sheet/{symbol}")
def server_get_balance_sheet(symbol: str, quarter: bool = False): 
    data = get_balance_sheet(symbol.upper(), quarter, DB_PATH)
    data = data.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)

@app.get("/get_cash_flow/{symbol}")
def server_get_cash_flow(symbol: str, quarter: bool = False): 
    data = get_cash_flow(symbol.upper(), quarter, DB_PATH)
    data = data.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)

@app.get("/get_news/{symbol}")
def server_get_news(symbol: str): 
    data = get_news(symbol.upper(), DB_PATH)
    data = data.to_pandas()
    data = data.to_json(orient="records", date_format="iso")
    return JSONResponse(content=data)


@app.get("/test")
def server_test():
    return "Hello!"



if __name__ == "__main__":
    # listen on all interfaces so your main PC can reach it
    
    uvicorn.run(app, host="0.0.0.0", port=8000)