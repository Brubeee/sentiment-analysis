# 📊 Financial Sentiment Analysis Pipeline

Analyze stock market sentiment using AI-powered analysis of news articles and social media posts. This pipeline correlates sentiment data with actual stock prices to generate investment insights.

---

## 🌟 Features

- ✅ **Multi-source Data**: News (GDELT) + Social Media (Reddit)
- ✅ **AI Sentiment Analysis**: FinBERT for financial news + RoBERTa for social media
- ✅ **Global Support**: US, Indian, European, Asian, and other international stocks
- ✅ **Price Correlation**: Links sentiment to actual stock performance
- ✅ **Flexible Time Periods**: Analyze from 1 month to 10+ years
- ✅ **Comprehensive Reports**: JSON output with detailed metrics

---

## 📁 Project Structure

```
├── main.py                    # Main orchestrator
├── config.py                  # All configuration settings
├── data_fetching.py          # Fetches price, news, social data
├── sentiment_analysis.py     # AI sentiment analysis
├── price_correlation.py      # Correlates sentiment with prices
├── requirements.txt          # Python dependencies
├── .env                      # API credentials (you create this)
└── results/                  # Output folder (auto-created)
```

---

## 🚀 Quick Start

### 1. Install (First Time Only)
See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed setup instructions.

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with Reddit credentials (optional)
```

### 2. Run Analysis
```bash
python main.py
```

That's it! Results appear in the `results/` folder.

---

## 🎯 How to Use

### Analyzing Different Stocks

Open `main.py` and find this section (around line 150):

```python
# Define tickers to analyze
tickers = ['LLY']
```

**Change to your desired stocks:**

```python
# Single stock
tickers = ['AAPL']

# Multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
```

Then run: `python main.py`

---

## 🌍 International Stock Support

This pipeline supports stocks from ALL major global exchanges!

### Supported Markets

| Exchange | Ticker Format | Example | Full Name |
|----------|---------------|---------|-----------|
| **US** (NYSE/NASDAQ) | `SYMBOL` | `AAPL` | Apple Inc |
| **India** (NSE) | `SYMBOL.NS` | `RELIANCE.NS` | Reliance Industries |
| **India** (BSE) | `SYMBOL.BO` | `TCS.BO` | Tata Consultancy |
| **UK** (London) | `SYMBOL.L` | `TSCO.L` | Tesco |
| **Japan** (Tokyo) | `SYMBOL.T` | `7203.T` | Toyota |
| **Hong Kong** | `SYMBOL.HK` | `0700.HK` | Tencent |
| **Canada** (Toronto) | `SYMBOL.TO` | `SHOP.TO` | Shopify |
| **Australia** | `SYMBOL.AX` | `CBA.AX` | CommBank |
| **Germany** | `SYMBOL.DE` | `VOW3.DE` | Volkswagen |

### Examples:

```python
# US stocks
tickers = ['AAPL', 'TSLA', 'NVDA']

# Indian stocks
tickers = ['RELIANCE.NS', 'TCS.BO', 'INFY.NS']

# European stocks
tickers = ['TSCO.L', 'VOW3.DE']

# Mixed international
tickers = ['AAPL', 'RELIANCE.NS', 'TSCO.L', '0700.HK']
```

### Finding Ticker Symbols

1. **Yahoo Finance**: Go to [finance.yahoo.com](https://finance.yahoo.com)
2. Search for your company
3. The ticker symbol is shown next to the company name
4. Use EXACTLY as shown (including the suffix like `.NS`, `.L`, etc.)

---

## ⏰ Changing Time Period

Open `config.py` and find this line (around line 15):

```python
LOOKBACK_YEARS = 2
```

### Common Time Periods:

```python
# Long-term analysis
LOOKBACK_YEARS = 10        # 10 years
LOOKBACK_YEARS = 5         # 5 years
LOOKBACK_YEARS = 2         # 2 years

# Medium-term analysis
LOOKBACK_YEARS = 1         # 1 year
LOOKBACK_YEARS = 0.5       # 6 months

# Short-term analysis
LOOKBACK_YEARS = 0.25      # 3 months
LOOKBACK_YEARS = 0.083     # 1 month
```

**Note**: Shorter periods = faster analysis but less reliable insights!

---

## 🔧 Advanced Configuration

All settings are in `config.py`. Here are the most useful ones:

### 1. Data Volume

```python
# How many articles to fetch per year
ARTICLES_PER_YEAR = 2000   # Default: 2000
                           # Increase for more data (slower)
                           # Decrease for faster analysis

# Reddit posts per subreddit
REDDIT_POST_LIMIT = 1000   # Per subreddit
```

### 2. Sentiment Weights

```python
# How much each source influences final sentiment
NEWS_WEIGHT = 0.6      # News contributes 60%
SOCIAL_WEIGHT = 0.4    # Social contributes 40%

# Adjust if you trust one source more
# Example: Trust news more
NEWS_WEIGHT = 0.8
SOCIAL_WEIGHT = 0.2
```

### 3. Processing Speed

```python
# Batch size for sentiment analysis
SENTIMENT_BATCH_SIZE = 32  # Default: 32
                          # Decrease to 16 or 8 if running out of memory
                          # Increase to 64 if you have powerful GPU
```

### 4. Analysis Windows

```python
# These auto-adjust based on LOOKBACK_YEARS
# But you can manually override:

SHORT_MOMENTUM_WINDOW = 30   # Days for short-term momentum
LONG_MOMENTUM_WINDOW = 180   # Days for long-term momentum
VOLATILITY_WINDOW = 30       # Days for volatility calculation
```

---

## 📊 Understanding Results

Results are saved as JSON files in `results/` folder.

### File Naming
- `TICKER_analysis_TIMESTAMP.json` - Individual stock results
- `SUMMARY_analysis_TIMESTAMP.json` - Combined results for multiple stocks

### Key Metrics Explained

**Sentiment Classification:**
- `SUPER POSITIVE` - Strong bullish sentiment (score > 0.6)
- `POSITIVE` - Bullish sentiment (score 0.2 to 0.6)
- `NEUTRAL` - Mixed sentiment (score -0.2 to 0.2)
- `NEGATIVE` - Bearish sentiment (score -0.6 to -0.2)
- `SUPER NEGATIVE` - Strong bearish sentiment (score < -0.6)

**Important Metrics:**
- `avg_sentiment`: Average sentiment score (-1 to +1)
- `sentiment_momentum`: Sentiment trend (positive = improving)
- `price_sentiment_correlation`: How well sentiment predicts price (-1 to +1)
- `total_return_pct`: Stock's total return in the period
- `annualized_return_pct`: Yearly average return

**Sentiment Period Performance:**
- Shows how stock performed during positive/negative sentiment days
- `win_rate`: % of days with positive returns

---

## 🎨 Customization Examples

### Example 1: Quick 3-Month Analysis
```python
# In config.py
LOOKBACK_YEARS = 0.25
ARTICLES_PER_YEAR = 1000  # Reduce for speed

# In main.py
tickers = ['AAPL']
```

### Example 2: Deep 10-Year Portfolio Analysis
```python
# In config.py
LOOKBACK_YEARS = 10
ARTICLES_PER_YEAR = 2000

# In main.py
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
```

### Example 3: Indian Market Focus
```python
# In config.py
LOOKBACK_YEARS = 2

# In main.py
tickers = [
    'RELIANCE.NS',   # Reliance Industries
    'TCS.BO',        # Tata Consultancy
    'INFY.NS',       # Infosys
    'HDFCBANK.NS',   # HDFC Bank
    'ITC.NS'         # ITC Limited
]
```

### Example 4: Tech Sector Comparison
```python
# In main.py
tickers = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google
    'NVDA',   # Nvidia
    'META',   # Meta
    'AMZN'    # Amazon
]
```

---

## 🐛 Common Issues & Solutions

### 1. "No price data found"
- **Problem**: Invalid ticker symbol
- **Solution**: Verify ticker on Yahoo Finance, include suffix (`.NS`, `.L`, etc.)

### 2. "No news articles found"
- **Problem**: Company too small or non-English news
- **Solution**: Try US stocks first, or increase `ARTICLES_PER_YEAR`

### 3. "No social media posts"
- **Problem**: Missing Reddit credentials or stock not discussed on Reddit
- **Solution**: Check `.env` file, or analysis will still work with just news data

### 4. Analysis takes too long
- **Solutions**:
  - Reduce `LOOKBACK_YEARS` in `config.py`
  - Decrease `ARTICLES_PER_YEAR`
  - Reduce `SENTIMENT_BATCH_SIZE`
  - Analyze fewer stocks at once

### 5. Out of memory errors
- **Solutions**:
  - Reduce `SENTIMENT_BATCH_SIZE` from 32 to 16 or 8
  - Analyze one stock at a time
  - Close other programs

### 6. "NaN values detected"
- **Problem**: Insufficient data for certain calculations
- **Solution**: Increase `LOOKBACK_YEARS` or try different stock

---

## 📖 How It Works

### Pipeline Flow:

1. **Data Fetching** (`data_fetching.py`)
   - Downloads stock prices from Yahoo Finance
   - Fetches news from GDELT (Global Database of Events, Language and Tone)
   - Gets social media posts from Reddit

2. **Sentiment Analysis** (`sentiment_analysis.py`)
   - Uses FinBERT AI model to analyze financial news
   - Uses RoBERTa AI model to analyze social media posts
   - Combines scores with weighted average

3. **Correlation Analysis** (`price_correlation.py`)
   - Merges sentiment data with price data
   - Calculates correlation between sentiment and returns
   - Measures momentum, volatility, and performance

4. **Classification** (`main.py`)
   - Combines all metrics into final score
   - Classifies as SUPER POSITIVE to SUPER NEGATIVE
   - Generates comprehensive report

---

## 🎓 Tips for Best Results

### ✅ Do:
- Start with well-known US stocks for testing
- Use 2+ years for reliable insights
- Analyze multiple stocks to compare
- Check correlation p-value (< 0.05 is significant)
- Look at sentiment momentum for trends

### ❌ Don't:
- Use extremely short periods (< 1 month)
- Analyze tiny companies with no news
- Rely solely on sentiment (use fundamental analysis too)
- Expect 100% accuracy (sentiment is one indicator)
- Ignore the confidence score

---

## 🔒 Privacy & Data

- **No personal data stored**: Only public financial data
- **API usage**: Complies with Reddit and GDELT terms
- **Local analysis**: All processing happens on your computer
- **Output**: Results saved locally in `results/` folder

---

## 📚 Files Reference

### Configuration Files
- **config.py** - All settings (time period, weights, batch sizes)
- **.env** - API credentials (you create this)

### Core Modules
- **main.py** - Main orchestrator, start here
- **data_fetching.py** - Downloads all data
- **sentiment_analysis.py** - AI sentiment scoring
- **price_correlation.py** - Statistical analysis

### Output
- **results/** - JSON files with analysis results

---

## 🚀 Performance Tips

### Faster Analysis:
- Reduce `LOOKBACK_YEARS`
- Decrease `ARTICLES_PER_YEAR`
- Lower `SENTIMENT_BATCH_SIZE`
- Use fewer tickers per run

### Better Accuracy:
- Increase `LOOKBACK_YEARS` (5-10 years recommended)
- Increase `ARTICLES_PER_YEAR` for more data
- Use stocks with high media coverage
- Check correlation p-values

### GPU Acceleration:
If you have NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
This can speed up sentiment analysis 3-5x!

---

## 🤝 Support

### Before Asking for Help:
1. Check [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
2. Verify ticker symbol on Yahoo Finance
3. Check `.env` file format
4. Try with just one stock first
5. Reduce `LOOKBACK_YEARS` to test

### Common Checks:
- Python version 3.9-3.11?
- All files in same folder?
- `.env` file created (if using Reddit)?
- Internet connection active?
- Sufficient disk space?

---

## 📝 Example Workflow

```bash
# 1. Setup (first time only)
pip install -r requirements.txt
# Create .env with Reddit credentials

# 2. Configure analysis
# Edit config.py -> Set LOOKBACK_YEARS = 2
# Edit main.py -> Set tickers = ['AAPL', 'MSFT']

# 3. Run analysis
python main.py

# 4. Check results
# Open results/SUMMARY_analysis_TIMESTAMP.json

# 5. Adjust and re-run
# Change tickers or time period
# Run again!
```

---

## 🎉 You're Ready!

Start with:
```python
# In main.py
tickers = ['AAPL']  # Start simple

# In config.py
LOOKBACK_YEARS = 2  # Medium-term analysis
```

Then run:
```bash
python main.py
```

**Happy Analyzing! 📈**

---

## ⚖️ Disclaimer

This tool is for educational and research purposes only. It does NOT provide financial advice. Always:
- Do your own research
- Consult financial advisors
- Consider multiple factors beyond sentiment
- Understand risks before investing

Past performance and sentiment do NOT guarantee future results.