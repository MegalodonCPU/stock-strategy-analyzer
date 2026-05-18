═══════════════════════════════════════════════════════
  STOCK STRATEGY ANALYZER — SETUP INSTRUCTIONS
═══════════════════════════════════════════════════════

FILES IN THIS FOLDER:
  app.py                    → Flask backend (Python)
  stock_strategy_widget.html → Frontend web interface
  README.txt                → This file

═══════════════════════════════════════════════════════
  STEP 1 — INSTALL DEPENDENCIES (first time only)
═══════════════════════════════════════════════════════

Open Command Prompt and run:

  pip install flask flask-cors pandas numpy yfinance scikit-learn xgboost tensorflow

If pip doesn't work try:
  py -m pip install flask flask-cors pandas numpy yfinance scikit-learn xgboost tensorflow

Note: TensorFlow is optional. If it fails to install the app
will still work using GBM only (no LSTM).

═══════════════════════════════════════════════════════
  STEP 2 — RUN THE APP
═══════════════════════════════════════════════════════

1. Open Command Prompt
2. Navigate to this folder:
     cd C:\Users\hrida\Desktop\Python_Projects
3. Run Flask:
     python app.py
4. Open Chrome and go to:
     http://localhost:8080

═══════════════════════════════════════════════════════
  STEP 3 — USE THE APP
═══════════════════════════════════════════════════════

1. Enter a ticker symbol (e.g. AAPL, GME, NVDA, TSLA)
2. Set your starting capital
3. Set start and end dates (use at least 3 years for ML)
4. Adjust trailing stop loss with the slider
5. Click Run Analysis

Note: ML model training takes 20-30 seconds. This is normal.
The app will show "Running analysis..." while it works.

═══════════════════════════════════════════════════════
  WHAT YOU GET
═══════════════════════════════════════════════════════

7 STRATEGIES:
  1. Buy & Hold              → baseline, just hold the stock
  2. Momentum                → MA50 vs MA200 crossover
  3. Bollinger Bands         → mean reversion on price bands
  4. Hybrid Bollinger Mom.   → both signals must agree to buy
  5. Trailing Stop Loss      → sells X% below peak price
  6. RSI                     → overbought/oversold signal
  7. Short / Bubble          → longs, goes cash, or shorts based on bubble score

AI RECOMMENDATION TAB:
  → GBM model predicts best strategy from current market features
  → LSTM model predicts from 30-day price sequences
  → Ensemble combines both for final recommendation
  → Bubble score shows crash risk (0-100)
  → Feature importance shows what drove the prediction

═══════════════════════════════════════════════════════
  TROUBLESHOOTING
═══════════════════════════════════════════════════════

Port 5000 blocked on Mac?
  → App uses port 8080 by default
  → Go to http://localhost:8080

"Could not connect to server"?
  → Make sure app.py is running in Command Prompt
  → Open the URL through Flask, not by double-clicking the HTML file

TensorFlow install fails?
  → App works without it (GBM only mode)
  → LSTM prediction will show N/A

Analysis takes too long?
  → Normal — ML training takes 20-30 seconds
  → Use a shorter date range to speed it up

═══════════════════════════════════════════════════════
  NOT FINANCIAL ADVICE
═══════════════════════════════════════════════════════

This tool is for educational purposes only.
Past performance does not guarantee future results.
Short selling involves unlimited downside risk.
Always consult a licensed financial advisor.

═══════════════════════════════════════════════════════
