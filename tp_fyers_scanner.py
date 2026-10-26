#!/usr/bin/env python3
"""
TrendPulse Breakout Scanner (Fyers data) -> Telegram alerts

Rules: Pine script jaise hi (ATR 21 x 6.3 trailing stop, Aqua = avg(VWMA100, trail),
Trigger = signal candle High/Low, SL = Aqua, Targets = Trigger +/- Gap x N).

Keys CODE MEIN MAT LIKHO. Environment variables use karo:
    FYERS_CLIENT_ID      (jaise 49XXXXXXXX-100)
    FYERS_ACCESS_TOKEN   (sirf eyJ... wala hissa)  -- manual mode / fallback
  Auto-login (roz token badalne ki zarurat nahi, ~15 din mein ek baar refresh token):
    FYERS_SECRET_KEY, FYERS_REFRESH_TOKEN, FYERS_PIN
    TG_TOKEN, TG_CHAT_ID

Install:  pip install fyers-apiv3 pandas numpy requests
Run:      python tp_fyers_scanner.py                 # market hours mein, due timeframes
          python tp_fyers_scanner.py --dry --force   # test, Telegram nahi bhejega
"""
import os, sys, json, math, time, argparse, hashlib, base64
import datetime as dt
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
