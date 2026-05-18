from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import math
import os

# ML imports
import ollama as ollama_client
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# PyTorch for LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    LSTM_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch available — using {DEVICE}")
except ImportError:
    LSTM_AVAILABLE = False
    DEVICE = None
    print("PyTorch not available — LSTM disabled, GBM only")

app = Flask(__name__)
CORS(app)

# ── data ──────────────────────────────────────────────────────────────────────

def get_data(ticker, start, end):
    """
    Downloads historical stock price data from Yahoo Finance.
    Pulls Close AND Volume — volume needed for bubble detection.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]]
    return df

# ── signals ───────────────────────────────────────────────────────────────────

def calculate_bollinger(df):
    """
    Calculates Bollinger Bands and generates a mean reversion signal.

    Signal logic:
        price <= Lower Band → 1  (buy)
        price >= Upper Band → 0  (sell)
        price in between   → nan → ffill
        days 1-19          → dropped
    """
    df["MB"]    = df["Close"].rolling(20).mean()
    df["STD"]   = df["Close"].rolling(20).std()
    df["Upper"] = df["MB"] + (2 * df["STD"])
    df["Lower"] = df["MB"] - (2 * df["STD"])
    df["Bollinger_Signal"] = np.where(
        df["Close"] <= df["Lower"], 1,
        np.where(df["Close"] >= df["Upper"], 0, np.nan)
    )
    df["Bollinger_Signal"] = df["Bollinger_Signal"].ffill()
    df.dropna(subset=["MB"], inplace=True)
    return df


def calculate_momentum(df):
    """
    Calculates moving averages and generates a momentum signal.

    Signal logic:
        MA50 > MA200 → 1  (buy — golden cross)
        MA50 < MA200 → 0  (sell — death cross)
        equal        → nan → ffill
        days 1-199   → dropped
    """
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Momentum_Signal"] = np.where(
        df["MA50"] > df["MA200"], 1,
        np.where(df["MA50"] < df["MA200"], 0, np.nan)
    )
    df["Momentum_Signal"] = df["Momentum_Signal"].ffill()
    df.dropna(subset=["MA200"], inplace=True)
    return df


def calculate_hybrid(df):
    """
    Combines Bollinger and Momentum signals.
    Requires BOTH to agree to buy. Sells if EITHER says sell.
    """
    df["Hybrid_Signal"] = np.where(
        (df["Momentum_Signal"] == 1) & (df["Bollinger_Signal"] == 1), 1,
        np.where(
            (df["Momentum_Signal"] == 0) | (df["Bollinger_Signal"] == 0), 0,
            np.nan
        )
    )
    df["Hybrid_Signal"] = df["Hybrid_Signal"].ffill()
    df.dropna(subset=["MA200", "MB"], inplace=True)
    return df


def calculate_trailing_stop(prices_df, stop_loss_pct):
    """
    Generates a trailing stop loss signal based on peak price tracking.
    Uses a for loop because each day depends on the running peak.

    How it works:
        - Buys on first day
        - Tracks peak price
        - Sells if price drops X% below peak
        - Stop only moves up, never down
    """
    signal     = []
    in_market  = False
    peak_price = 0

    for price in prices_df["Close"]:
        if not in_market:
            in_market  = True
            peak_price = price
            signal.append(1)
        else:
            if price > peak_price:
                peak_price = price
            stop_price = peak_price * (1 - stop_loss_pct / 100)
            if price <= stop_price:
                in_market  = False
                peak_price = 0
                signal.append(0)
            else:
                signal.append(1)

    prices_df["Trailing_Signal"] = signal
    return prices_df


def calculate_rsi(df, period=14):
    """
    Calculates RSI and generates an overbought/oversold signal.

    Signal logic:
        RSI < 30  → 1  (oversold → buy)
        RSI > 70  → 0  (overbought → sell)
        between   → nan → ffill
        days 1-14 → dropped
    """
    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Signal"] = np.where(
        df["RSI"] < 30, 1,
        np.where(df["RSI"] > 70, 0, np.nan)
    )
    df["RSI_Signal"] = df["RSI_Signal"].ffill()
    df.dropna(subset=["RSI"], inplace=True)
    return df


def calculate_bubble_and_short(df):
    """
    Detects market bubbles and generates a long/cash/short signal.

    Bubble score (0-100):
        Price deviation from MA200  → max 30 pts
        Extended RSI overbought     → max 25 pts
        High volatility while rising → max 25 pts
        Volume spike                → max 20 pts

    Signal logic:
        Bubble_Score > 85 → -1  (extreme bubble → short)
        Bubble_Score > 70 →  0  (mild bubble → stay out)
        RSI < 30          →  1  (oversold → buy)
        else              → nan → ffill

    Short borrowing cost = 1% annual / 252 trading days
    """
    deviation    = (df["Close"] - df["MA200"]) / df["MA200"] * 100
    rsi_extended = (df["RSI"] > 70).rolling(10).sum()
    vol          = df["Close"].pct_change().rolling(20).std() * 100
    rising       = df["Close"] > df["MA50"]
    vol_spike    = df["Volume"] / df["Volume"].rolling(20).mean()

    df["RSI_Extended"] = rsi_extended
    df["Volume_Spike"] = vol_spike

    score = pd.Series(0.0, index=df.index)
    score += np.clip(deviation / 2, 0, 30)
    score += np.clip(rsi_extended * 2.5, 0, 25)
    score += np.where(rising & (vol > 2), np.clip(vol * 5, 0, 25), 0)
    score += np.clip((vol_spike - 1) * 10, 0, 20)

    df["Bubble_Score"]    = score
    df["Price_Deviation"] = deviation

    df["Short_Signal"] = np.where(
        score > 85, -1,
        np.where(score > 70, 0,
            np.where(df["RSI"] < 30, 1, np.nan))
    )
    df["Short_Signal"] = df["Short_Signal"].ffill()
    df["Short_Cost"]   = np.where(df["Short_Signal"].shift(1) == -1, 0.01 / 252, 0)
    return df

# ── returns ───────────────────────────────────────────────────────────────────

def calculate_returns(df, capital):
    """
    Calculates daily and cumulative returns for all seven strategies.
    shift(1) prevents lookahead bias.

    Short signal math:
        signal=1  × +3% = +3%  (long profit)
        signal=0  × +3% =  0%  (cash)
        signal=-1 × +3% = -3%  (short loss)
        signal=-1 × -3% = +3%  (short profit)
    """
    df["Market_Returns"]    = df["Close"].pct_change()
    df["Momentum_Returns"]  = df["Momentum_Signal"].shift(1)  * df["Market_Returns"]
    df["Bollinger_Returns"] = df["Bollinger_Signal"].shift(1) * df["Market_Returns"]
    df["Hybrid_Returns"]    = df["Hybrid_Signal"].shift(1)    * df["Market_Returns"]
    df["Trailing_Returns"]  = df["Trailing_Signal"].shift(1)  * df["Market_Returns"]
    df["RSI_Returns"]       = df["RSI_Signal"].shift(1)       * df["Market_Returns"]
    df["Short_Returns"]     = (df["Short_Signal"].shift(1) * df["Market_Returns"]) - df["Short_Cost"]

    df["Market_Cumulative"]    = (1 + df["Market_Returns"]).cumprod()    * capital
    df["Momentum_Cumulative"]  = (1 + df["Momentum_Returns"]).cumprod()  * capital
    df["Bollinger_Cumulative"] = (1 + df["Bollinger_Returns"]).cumprod() * capital
    df["Hybrid_Cumulative"]    = (1 + df["Hybrid_Returns"]).cumprod()    * capital
    df["Trailing_Cumulative"]  = (1 + df["Trailing_Returns"]).cumprod()  * capital
    df["RSI_Cumulative"]       = (1 + df["RSI_Returns"]).cumprod()       * capital
    df["Short_Cumulative"]     = (1 + df["Short_Returns"]).cumprod()     * capital

    df["Margin_Call"] = df["Short_Cumulative"] < capital * 0.5
    return df

# ── helpers ───────────────────────────────────────────────────────────────────

def clean(val):
    """
    Converts numpy types to Python native types and replaces NaN with None.
    Flask jsonify cannot serialize numpy float32/float64 or NaN directly.
    """
    if val is None:
        return None
    if hasattr(val, 'item'):        # converts numpy scalar → Python native
        val = val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    return round(float(val), 2)


def safe_float(val):
    """Safely converts any numeric value to a Python float."""
    if val is None:
        return 0.0
    if hasattr(val, 'item'):
        val = val.item()
    return float(val)

# ── metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(df, col_returns, col_signal, capital):
    """
    Calculates standard performance metrics for a single strategy.
    All values explicitly converted to Python float to avoid
    JSON serialization errors with numpy float32/float64.
    """
    final_value   = safe_float(df[col_returns.replace("Returns", "Cumulative")].iloc[-1])
    delta         = final_value - capital
    pct_return    = (delta / capital) * 100

    daily_returns = df[col_returns].dropna()
    sharpe = 0.0
    if safe_float(daily_returns.std()) > 0:
        sharpe = safe_float((daily_returns.mean() / daily_returns.std()) * (252 ** 0.5))

    cumulative   = df[col_returns.replace("Returns", "Cumulative")]
    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_drawdown = safe_float(drawdown.min()) * 100

    if col_signal and col_signal in df.columns:
        time_in_market = safe_float((df[col_signal] == 1).mean()) * 100
    else:
        time_in_market = 100.0

    return {
        "final_value"   : round(final_value, 2),
        "delta"         : round(delta, 2),
        "pct_return"    : round(pct_return, 2),
        "sharpe"        : round(sharpe, 2),
        "max_drawdown"  : round(max_drawdown, 2),
        "time_in_market": round(time_in_market, 1)
    }


def calculate_short_metrics(df, capital):
    """
    Extended metrics for short/bubble strategy.
    Adds time breakdown, borrowing costs, margin calls.
    """
    standard = calculate_metrics(df, "Short_Returns", None, capital)

    standard["time_long"]      = round(safe_float((df["Short_Signal"] == 1).mean())  * 100, 1)
    standard["time_cash"]      = round(safe_float((df["Short_Signal"] == 0).mean())  * 100, 1)
    standard["time_short"]     = round(safe_float((df["Short_Signal"] == -1).mean()) * 100, 1)
    standard["borrowing_cost"] = round(safe_float(df["Short_Cost"].sum()) * capital, 2)
    standard["margin_calls"]   = int(df["Margin_Call"].sum())
    standard["bubble_score"]   = round(safe_float(df["Bubble_Score"].iloc[-1]), 1)

    return standard

# ── ML model ──────────────────────────────────────────────────────────────────

def build_feature_matrix(df):
    """
    Builds the feature matrix for the ML models.

    Features:
        RSI              → overbought/oversold level
        MA50             → short term price trend
        MA200            → long term price trend
        STD              → price volatility
        Price_Deviation  → how far price is from MA200
        Bubble_Score     → composite bubble detection score
        RSI_Extended     → days RSI has been overbought
        Volume_Spike     → unusual volume vs 20 day average
        Recent_5         → 5 day price return
        Recent_20        → 20 day price return
    """
    features = pd.DataFrame(index=df.index)
    features["RSI"]             = df["RSI"]
    features["MA50"]            = df["MA50"]
    features["MA200"]           = df["MA200"]
    features["STD"]             = df["STD"]
    features["Price_Deviation"] = df["Price_Deviation"]
    features["Bubble_Score"]    = df["Bubble_Score"]
    features["RSI_Extended"]    = df["RSI_Extended"]
    features["Volume_Spike"]    = df["Volume_Spike"]
    features["Recent_5"]        = df["Close"].pct_change(5)  * 100
    features["Recent_20"]       = df["Close"].pct_change(20) * 100
    return features.dropna()


def label_best_strategy(df, features, capital, forward_days=30):
    """
    Labels which strategy performed best in the next forward_days days.
    This is the target variable for ML training.
    """
    strategies = {
        "market"   : "Market_Cumulative",
        "momentum" : "Momentum_Cumulative",
        "bollinger": "Bollinger_Cumulative",
        "hybrid"   : "Hybrid_Cumulative",
        "trailing" : "Trailing_Cumulative",
        "rsi"      : "RSI_Cumulative",
        "short"    : "Short_Cumulative",
    }

    labels      = []
    feature_idx = features.index

    for i in range(len(feature_idx) - forward_days):
        current_date  = feature_idx[i]
        future_date   = feature_idx[min(i + forward_days, len(feature_idx) - 1)]
        best_strategy = "market"
        best_return   = -999

        for name, col in strategies.items():
            if col not in df.columns:
                continue
            try:
                current_val    = safe_float(df.loc[current_date, col])
                future_val     = safe_float(df.loc[future_date, col])
                forward_return = (future_val - current_val) / current_val * 100
                if forward_return > best_return:
                    best_return   = forward_return
                    best_strategy = name
            except:
                continue

        labels.append(best_strategy)

    return labels


def train_gbm(X_train, y_train):
    """
    Trains a Gradient Boosting Classifier on historical market features.

    GBM is ideal for tabular financial data:
        - Handles non-linear relationships
        - Resistant to overfitting with shallow trees
        - Provides feature importance scores
        - Fast to train and predict
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model    = GradientBoostingClassifier(
        n_estimators  = 100,
        learning_rate = 0.1,
        max_depth     = 3,
        random_state  = 42
    )
    model.fit(X_scaled, y_train)
    return model, scaler

# ── PyTorch LSTM ──────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    PyTorch LSTM neural network for sequential pattern recognition.

    Architecture:
        Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(n_classes)

    LSTM captures sequential patterns in price data:
        - Remembers patterns over sequence_length days
        - Captures momentum and regime changes
        - Complements GBM which only sees current snapshot
        - Runs on RTX 4060 GPU via CUDA for fast training
    """
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, n_classes=7):
        super(LSTMModel, self).__init__()
        self.lstm1    = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2    = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc       = nn.Linear(hidden_size2, n_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.dropout1(out)
        out, _ = self.lstm2(out)
        out    = self.dropout2(out)
        out    = self.fc(out[:, -1, :])    # take last timestep
        return out


def train_lstm_pytorch(X_train, y_train, strategy_classes, sequence_length=30):
    """
    Trains PyTorch LSTM on sequences of past market data.
    Automatically uses RTX 4060 GPU via CUDA.

    Parameters:
        X_train         (array) : feature matrix [n_samples, n_features]
        y_train         (array) : strategy labels (strings)
        strategy_classes (list) : all possible strategy names
        sequence_length (int)   : how many past days LSTM looks at

    Returns:
        model  : trained PyTorch LSTM
        scaler : fitted StandardScaler
        le     : fitted LabelEncoder
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    le = LabelEncoder()
    le.fit(strategy_classes)
    y_encoded = le.transform(y_train)

    sequences  = []
    seq_labels = []
    for i in range(sequence_length, len(X_scaled)):
        sequences.append(X_scaled[i-sequence_length:i])
        seq_labels.append(y_encoded[i])

    if len(sequences) < 10:
        return None, None, None

    X_seq = torch.FloatTensor(np.array(sequences)).to(DEVICE)
    y_seq = torch.LongTensor(np.array(seq_labels)).to(DEVICE)

    dataset    = TensorDataset(X_seq, y_seq)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    n_features = X_train.shape[1]
    n_classes  = len(strategy_classes)
    model      = LSTMModel(n_features, n_classes=n_classes).to(DEVICE)
    optimizer  = optim.Adam(model.parameters(), lr=0.001)
    criterion  = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(20):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, scaler, le


def run_ml_prediction(df, features, labels, capital):
    """
    Trains GBM and LSTM and generates ensemble prediction.

    Ensemble:
        60% GBM weight + 40% LSTM weight = final prediction
        Both agree → higher confidence
        Disagree   → lower confidence
    """
    strategy_classes = ["market", "momentum", "bollinger", "hybrid",
                        "trailing", "rsi", "short"]
    strategy_names = {
        "market"   : "Buy & Hold",
        "momentum" : "Momentum",
        "bollinger": "Bollinger Bands",
        "hybrid"   : "Hybrid Bollinger Momentum",
        "trailing" : "Trailing Stop Loss",
        "rsi"      : "RSI",
        "short"    : "Short / Bubble",
    }

    feature_rows = features.iloc[:len(labels)]
    X = feature_rows.values
    y = np.array(labels)

    if len(X) < 60:
        return {"error": "Not enough data for ML. Use a longer date range (3+ years)."}

    # ── GBM ───────────────────────────────────────────────────────────────────
    gbm_model, gbm_scaler = train_gbm(X, y)

    current_features = features.iloc[-1:].values
    current_scaled   = gbm_scaler.transform(current_features)
    gbm_proba        = gbm_model.predict_proba(current_scaled)[0]
    gbm_classes      = gbm_model.classes_
    gbm_pred         = str(gbm_classes[np.argmax(gbm_proba)])
    gbm_confidence   = round(float(np.max(gbm_proba)) * 100, 1)

    importance   = dict(zip(feature_rows.columns, gbm_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_pred       = None
    lstm_confidence = None
    ensemble_pred   = gbm_pred
    ensemble_conf   = gbm_confidence
    sequence_length = 30

    if LSTM_AVAILABLE and len(X) > sequence_length + 10:
        try:
            lstm_model, lstm_scaler, le = train_lstm_pytorch(
                X, y, strategy_classes, sequence_length
            )

            if lstm_model is not None:
                X_scaled_full = lstm_scaler.transform(X)
                last_seq      = X_scaled_full[-sequence_length:]
                last_tensor   = torch.FloatTensor(last_seq).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    lstm_output = lstm_model(last_tensor)
                    lstm_proba  = torch.softmax(lstm_output, dim=1).cpu().numpy()[0]

                lstm_pred_idx   = int(np.argmax(lstm_proba))
                lstm_pred       = str(le.inverse_transform([lstm_pred_idx])[0])
                lstm_confidence = round(float(np.max(lstm_proba)) * 100, 1)

                gbm_proba_full  = np.zeros(len(strategy_classes))
                lstm_proba_full = np.zeros(len(strategy_classes))

                for i, cls in enumerate(gbm_classes):
                    if cls in strategy_classes:
                        gbm_proba_full[strategy_classes.index(cls)] = float(gbm_proba[i])

                for i, cls in enumerate(le.classes_):
                    if cls in strategy_classes:
                        lstm_proba_full[strategy_classes.index(cls)] = float(lstm_proba[i])

                ensemble_proba = 0.6 * gbm_proba_full + 0.4 * lstm_proba_full
                ensemble_idx   = int(np.argmax(ensemble_proba))
                ensemble_pred  = str(strategy_classes[ensemble_idx])
                ensemble_conf  = round(float(np.max(ensemble_proba)) * 100, 1)

        except Exception as e:
            print(f"LSTM training failed: {e} — using GBM only")

    # ── current conditions ────────────────────────────────────────────────────
    current      = features.iloc[-1]
    bubble_score = round(safe_float(df["Bubble_Score"].iloc[-1]), 1)
    current_rsi  = round(safe_float(current["RSI"]), 1)
    current_ma50 = safe_float(current["MA50"])
    current_ma200= safe_float(current["MA200"])

    if bubble_score > 85:
        regime = "Extreme bubble — high crash risk"
    elif bubble_score > 70:
        regime = "Bubble forming — elevated risk"
    elif current_rsi < 30:
        regime = "Oversold — potential buying opportunity"
    elif current_rsi > 70:
        regime = "Overbought — potential selling opportunity"
    elif current_ma50 > current_ma200:
        regime = "Upward trend — momentum positive"
    else:
        regime = "Downward trend — momentum negative"

    return {
        "recommendation"    : strategy_names.get(ensemble_pred, ensemble_pred),
        "recommendation_key": ensemble_pred,
        "confidence"        : ensemble_conf,
        "gbm_prediction"    : strategy_names.get(gbm_pred, gbm_pred),
        "gbm_confidence"    : gbm_confidence,
        "lstm_prediction"   : strategy_names.get(lstm_pred, lstm_pred) if lstm_pred else "N/A",
        "lstm_confidence"   : lstm_confidence if lstm_confidence else "N/A",
        "regime"            : regime,
        "bubble_score"      : bubble_score,
        "current_rsi"       : current_rsi,
        "current_ma_trend"  : "Upward" if current_ma50 > current_ma200 else "Downward",
        "current_volatility": round(safe_float(current["STD"]), 2),
        "top_features"      : [{"name": str(k), "importance": round(float(v) * 100, 1)}
                               for k, v in top_features],
        "models_used"       : f"GBM + LSTM Ensemble (GPU: {DEVICE})" if lstm_pred else "GBM Only",
    }

# ── master function ───────────────────────────────────────────────────────────

def run_analysis(ticker, start, end, capital, stop_loss_pct):
    """
    Master function that runs the full analysis pipeline.

    Order matters:
        1.  get_data()
        2.  calculate_bollinger()
        3.  calculate_momentum()
        4.  calculate_hybrid()       → needs bollinger + momentum
        5.  calculate_trailing_stop()
        6.  calculate_rsi()
        7.  calculate_bubble_and_short() → needs RSI + MA columns
        8.  calculate_returns()      → needs all signals
        9.  build_feature_matrix()
        10. label_best_strategy()    → needs cumulative returns
        11. run_ml_prediction()
    """
    df = get_data(ticker, start, end)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df = calculate_bollinger(df)
    df = calculate_momentum(df)
    df = calculate_hybrid(df)
    df = calculate_trailing_stop(df, stop_loss_pct)
    df = calculate_rsi(df)
    df = calculate_bubble_and_short(df)
    df = calculate_returns(df, capital)

    features  = build_feature_matrix(df)
    labels    = label_best_strategy(df, features, capital, forward_days=30)
    ml_result = run_ml_prediction(df, features, labels, capital)

    dates = df.index.strftime("%Y-%m-%d").tolist()

    series = {
        "market"   : [clean(v) for v in df["Market_Cumulative"].tolist()],
        "momentum" : [clean(v) for v in df["Momentum_Cumulative"].tolist()],
        "bollinger": [clean(v) for v in df["Bollinger_Cumulative"].tolist()],
        "hybrid"   : [clean(v) for v in df["Hybrid_Cumulative"].tolist()],
        "trailing" : [clean(v) for v in df["Trailing_Cumulative"].tolist()],
        "rsi"      : [clean(v) for v in df["RSI_Cumulative"].tolist()],
        "short"    : [clean(v) for v in df["Short_Cumulative"].tolist()],
    }

    metrics = {
        "market"   : calculate_metrics(df, "Market_Returns",    None,               capital),
        "momentum" : calculate_metrics(df, "Momentum_Returns",  "Momentum_Signal",  capital),
        "bollinger": calculate_metrics(df, "Bollinger_Returns", "Bollinger_Signal", capital),
        "hybrid"   : calculate_metrics(df, "Hybrid_Returns",    "Hybrid_Signal",    capital),
        "trailing" : calculate_metrics(df, "Trailing_Returns",  "Trailing_Signal",  capital),
        "rsi"      : calculate_metrics(df, "RSI_Returns",       "RSI_Signal",       capital),
        "short"    : calculate_short_metrics(df, capital),
    }

    bh_return = metrics["market"]["pct_return"]
    for key in metrics:
        metrics[key]["vs_bh"] = round(metrics[key]["pct_return"] - bh_return, 2)

    return {
        "ticker" : ticker,
        "start"  : start,
        "end"    : end,
        "capital": capital,
        "dates"  : dates,
        "series" : series,
        "metrics": metrics,
        "ml"     : ml_result,
    }

# ── flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serves the main HTML widget"""
    response = send_from_directory(".", "stock_strategy_widget.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main API endpoint. Receives ticker/dates/capital/stop_loss from frontend.
    Runs full analysis pipeline including ML prediction.
    Returns JSON with series, metrics and ML recommendation.
    """
    try:
        data          = request.get_json()
        ticker        = data.get("ticker", "AAPL").upper()
        start         = data.get("start", "2020-01-01")
        end           = data.get("end", "2026-04-24")
        capital       = float(data.get("capital", 10000))
        stop_loss_pct = int(data.get("stop_loss_pct", 5))

        result = run_analysis(ticker, start, end, capital, stop_loss_pct)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "lstm"  : LSTM_AVAILABLE,
        "device": str(DEVICE) if DEVICE else "cpu"
    })

# ── stockbot ──────────────────────────────────────────────────────────────────

@app.route("/stockbot", methods=["POST"])
def stockbot():
    """
    StockBot endpoint. Receives the current analysis context + user message.
    Passes both to Llama3 via Ollama and returns a plain English response.
    """
    try:
        data    = request.get_json()
        message = data.get("message", "")
        context = data.get("context", {})

        # Build a context summary from the analysis results
        ml      = context.get("ml", {})
        metrics = context.get("metrics", {})
        ticker  = context.get("ticker", "the stock")

        context_prompt = f"""
You are StockBot, a helpful AI assistant built into a stock strategy analyzer app.
The user has just run an analysis on {ticker}. Here are the results:

ML Recommendation: {ml.get('recommendation', 'N/A')}
Confidence: {ml.get('confidence', 'N/A')}%
Market Regime: {ml.get('regime', 'N/A')}
Bubble Score: {ml.get('bubble_score', 'N/A')} / 100
Current RSI: {ml.get('current_rsi', 'N/A')}
MA Trend: {ml.get('current_ma_trend', 'N/A')}
Current Volatility: {ml.get('current_volatility', 'N/A')}%
GBM Prediction: {ml.get('gbm_prediction', 'N/A')} ({ml.get('gbm_confidence', 'N/A')}% confidence)
LSTM Prediction: {ml.get('lstm_prediction', 'N/A')} ({ml.get('lstm_confidence', 'N/A')}% confidence)
Models Used: {ml.get('models_used', 'N/A')}

Strategy Performance (historical backtest):
- Buy & Hold: {metrics.get('market', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('market', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('market', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('market', {}).get('time_in_market', 'N/A')}%
- Momentum: {metrics.get('momentum', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('momentum', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('momentum', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('momentum', {}).get('time_in_market', 'N/A')}%
- Bollinger Bands: {metrics.get('bollinger', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('bollinger', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('bollinger', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('bollinger', {}).get('time_in_market', 'N/A')}%
- Hybrid Bollinger Momentum: {metrics.get('hybrid', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('hybrid', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('hybrid', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('hybrid', {}).get('time_in_market', 'N/A')}%
- Trailing Stop: {metrics.get('trailing', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('trailing', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('trailing', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('trailing', {}).get('time_in_market', 'N/A')}%
- RSI: {metrics.get('rsi', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('rsi', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('rsi', {}).get('max_drawdown', 'N/A')}%, Time in Market {metrics.get('rsi', {}).get('time_in_market', 'N/A')}%
- Short/Bubble: {metrics.get('short', {}).get('pct_return', 'N/A')}% return, Sharpe {metrics.get('short', {}).get('sharpe', 'N/A')}, Max Drawdown {metrics.get('short', {}).get('max_drawdown', 'N/A')}%

CRITICAL INSTRUCTIONS:
1. Buy & Hold is a valid strategy the model can recommend. It means simply buying the stock and holding it for the entire period with no trading.
2. When explaining why a strategy was recommended, you MUST compare it against Buy & Hold specifically. Every investor's default option is to just hold, so you must explain why the recommended strategy is better than simply holding — or acknowledge if Buy & Hold actually outperformed historically and explain why the AI still recommends something else based on FORWARD-LOOKING conditions.
3. The AI recommendation is based on CURRENT market conditions and what the models predict will work best in the NEXT 30 days. Historical returns show what worked in the PAST. These can differ. If the recommended strategy has lower historical returns than Buy & Hold, explain that the AI is optimizing for the future, not the past.
4. Use the Sharpe ratio and max drawdown to explain risk-adjusted performance. A strategy with lower returns but a better Sharpe ratio and smaller drawdown may be preferable for risk management.
5. Be specific with numbers from the data above. Do not make up numbers.
6. Be concise, friendly, and helpful. Keep responses under 200 words unless the user asks for detail.
"""

        response = ollama_client.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user",   "content": message}
            ]
        )

        reply = response["message"]["content"]
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Stock Strategy Analyzer...")
    print(f"LSTM available : {LSTM_AVAILABLE}")
    print(f"Device         : {DEVICE}")
    print("Open your browser at: http://localhost:8080")
    app.run(debug=True, port=5000, host='127.0.0.1')
