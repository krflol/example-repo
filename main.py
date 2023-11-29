
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, BackgroundTasks,APIRouter
import subprocess
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import pandas as pd
import sqlite3
from multiprocessing import Process, Queue
import threading

try:
    import torch
except ImportError:
    subprocess.check_call(["pip", "install", "torch"])
    import torch
import fastapi.middleware.cors
from routers.calvin import datarecorder, maximized_agent,bnn_trader, pporouter,dqnrouter #sonofcalvinroute xploiter,explorer,policyexplorer, get_top_100,pporouter
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import Process
current_timestamp = None
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3040",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import subprocess



app.include_router(pporouter.router, prefix="/pporouter", tags=["pporouter"])
#to run the server on port 3040, use the following command: uvicorn main:app --reload --port 3040
#to run it withouthout info logging, use the following command: uvicorn main:app --reload --port 3040 --log-level critical






