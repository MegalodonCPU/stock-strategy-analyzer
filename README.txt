# Stock Strategy Analyzer

An AI-powered web application that backtests seven trading strategies against historical market data and uses a machine learning ensemble to recommend the optimal strategy for the next 30 days based on current market conditions.

Built with Flask, PyTorch, and scikit-learn. Features StockBot — a conversational AI assistant that explains every recommendation in plain English.

## Features

- **Seven trading strategies** backtested side by side: Buy & Hold, Momentum, Bollinger Bands, Hybrid Bollinger-Momentum, Trailing Stop Loss, RSI, and Short/Bubble
- **ML strategy recommendation** powered by a Gradient Boosted Machine (GBM) + LSTM neural network ensemble
- **Bubble detection** that scores market overheating on a 0-100 scale using price deviation, RSI extension, volatility, and volume signals
- **Realistic short selling** with 150% margin requirements and knock-out barrier liquidation (mirrors real brokerage mechanics)
- **StockBot AI assistant** that explains recommendations, compares strategies against Buy & Hold, and answers questions about your analysis
- **Interactive performance chart** with per-strategy crosshair tracking
- **Risk metrics** for every strategy: total return, Sharpe ratio, maximum drawdown, and time in market
- **Light and dark themes**

## How It Works

The analyzer pulls historical price data for any US-listed ticker, then runs each of the seven strategies through a backtest starting from a configurable capital amount. Each strategy generates buy, sell, hold, or short signals based on its own logic.

The ML ensemble analyzes current technical indicators (RSI, moving average trends, volatility, bubble score, and market regime) and predicts which strategy is best positioned for the next 30 days. Unlike a simple historical ranking, the model optimizes for forward-looking conditions — it can recommend Buy & Hold during steady uptrends, defensive cash positions during overheating, or active shorting during extreme bubbles.

StockBot wraps the entire analysis in a conversational layer. Ask it why a strategy was recommended, what RSI means, or how the current market regime affects your stock, and it responds using the actual numbers from your analysis.

## Tech Stack

- **Backend:** Flask (Python)
- **ML:** PyTorch (LSTM), scikit-learn (GBM), trained on GPU when available
- **Data:** yfinance for historical market data
- **AI Assistant:** Llama 3.1 via Groq API
- **Frontend:** Vanilla JavaScript with HTML5 Canvas charting

## Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (PyTorch does not yet support 3.13+)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/MegalodonCPU/stock-strategy-analyzer.git
cd stock-strategy-analyzer
pip install -r requirements.txt
```

### Configuration

Set your Groq API key as an environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="your_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY=your_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"
```

### Running

```bash
python app.py
```

Then open your browser to `http://localhost:5000`.

## Usage

1. Enter a stock ticker (e.g., AAPL, TSLA, NVDA)
2. Set your starting capital, date range, and trailing stop percentage
3. Click **Run Analysis**
4. Review the performance chart and strategy cards
5. Check the AI recommendation and click **Why?** for a detailed explanation
6. Ask StockBot any follow-up questions

## Strategy Overview

| Strategy | Logic |
|----------|-------|
| Buy & Hold | Buy on day one, hold the entire period |
| Momentum | Follow the trend using moving average crossovers |
| Bollinger Bands | Buy at the lower band, sell at the upper band |
| Hybrid Bollinger-Momentum | Combine mean reversion with trend confirmation |
| Trailing Stop Loss | Ride gains, exit on a percentage pullback from the peak |
| RSI | Buy oversold conditions, sell overbought |
| Short/Bubble | Stay long by default, go to cash on overheating, short on extreme bubbles |

## Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Past performance does not guarantee future results. Always consult a licensed financial advisor before making investment decisions.

## License

MIT
