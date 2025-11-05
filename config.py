"""
Configuration file for the 10-Year Financial Sentiment Analysis Pipeline.
All configurable parameters are defined here.
"""

import os
from datetime import datetime, timedelta

# ============================================================================
# TIME PERIOD CONFIGURATION
# ============================================================================

# Number of years to look back for historical analysis (supports float values)
# Examples: 
#   10 = 10 years
#   0.5 = 6 months
#   0.25 = 3 months
#   0.083 = 1 month (approximately)
LOOKBACK_YEARS = 2

# Calculate date range
END_DATE = datetime.now()
# Convert years to days (use exact calculation)
LOOKBACK_DAYS = int(LOOKBACK_YEARS * 365.25)  # Account for leap years
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)

print(f"[CONFIG] Analysis period: {START_DATE.date()} to {END_DATE.date()}")
print(f"[CONFIG] Lookback: {LOOKBACK_YEARS} years ({LOOKBACK_DAYS} days)")

# ============================================================================
# DATA FETCHING CONFIGURATION
# ============================================================================

# GDELT News Fetching Strategy
# This limits the number of articles per year to avoid overwhelming data
ARTICLES_PER_YEAR = 2000  # 2000 articles/year

# Calculate total articles based on actual lookback period
TOTAL_ARTICLES_TARGET = int(ARTICLES_PER_YEAR * LOOKBACK_YEARS)
print(f"[CONFIG] Target articles: {TOTAL_ARTICLES_TARGET}")

# Reddit Configuration
REDDIT_SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
REDDIT_POST_LIMIT = int(1000 * LOOKBACK_YEARS)  # Scale posts by lookback period
print(f"[CONFIG] Reddit posts per subreddit: {REDDIT_POST_LIMIT}")

# ============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# ============================================================================

# Sentiment model names
# FinBERT is specifically trained on financial text
NEWS_SENTIMENT_MODEL = 'ProsusAI/finbert'

# Twitter-RoBERTa is trained on social media text
SOCIAL_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# Sentiment weights for final combined score
NEWS_WEIGHT = 0.6      # News sentiment contributes 60%
SOCIAL_WEIGHT = 0.4    # Social sentiment contributes 40%

# Batch processing
SENTIMENT_BATCH_SIZE = 32  # Process 32 texts at a time

# ============================================================================
# API KEYS (from .env file)
# ============================================================================

# Reddit API credentials (hardcoded)
REDDIT_CLIENT_ID = 'ksKUj28ppl8OohZpurvMAQ'  # Your actual ID
REDDIT_CLIENT_SECRET = 'V9-aewb6y3IKWQm63FHdNd3VRchBdQ'  # Your actual secret
REDDIT_USER_AGENT = 'SentimentAnalysisPipeline/1.0'
# ============================================================================
# CORRELATION ANALYSIS CONFIGURATION
# ============================================================================

# Momentum calculation periods (in days)
# Automatically scale based on lookback period
if LOOKBACK_YEARS >= 1.0:
    SHORT_MOMENTUM_WINDOW = 30   # 1 month
    LONG_MOMENTUM_WINDOW = 180   # 6 months
elif LOOKBACK_YEARS >= 0.5:
    SHORT_MOMENTUM_WINDOW = 7    # 1 week
    LONG_MOMENTUM_WINDOW = 30    # 1 month
else:
    SHORT_MOMENTUM_WINDOW = 5    # 5 days
    LONG_MOMENTUM_WINDOW = 15    # 15 days

print(f"[CONFIG] Momentum windows: short={SHORT_MOMENTUM_WINDOW} days, long={LONG_MOMENTUM_WINDOW} days")

# Volatility calculation
VOLATILITY_WINDOW = max(5, int(30 * LOOKBACK_YEARS))  # Scale but minimum 5 days
print(f"[CONFIG] Volatility window: {VOLATILITY_WINDOW} days")

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Results directory
RESULTS_DIR = 'results'

# Cache directory for downloaded models
CACHE_DIR = './model_cache'

# ============================================================================
# GDELT QUERY CONFIGURATION
# ============================================================================

# GDELT query keywords to ensure we get financial articles
FINANCIAL_KEYWORDS = ['stock', 'market', 'earnings', 'finance', 'investor', 'trading']

# Maximum number of articles to fetch per GDELT query (safety limit)
MAX_GDELT_ARTICLES_PER_QUERY = 5000

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that all required configuration is present."""
    errors = []
    
    if not REDDIT_CLIENT_ID:
        errors.append("REDDIT_CLIENT_ID not found in environment variables")
    if not REDDIT_CLIENT_SECRET:
        errors.append("REDDIT_CLIENT_SECRET not found in environment variables")
    
    # Validate lookback period
    if LOOKBACK_YEARS <= 0:
        errors.append(f"LOOKBACK_YEARS must be positive (got {LOOKBACK_YEARS})")
    
    if LOOKBACK_YEARS < 0.08:  # Less than 1 month
        print(f"⚠️  WARNING: Very short lookback period ({LOOKBACK_YEARS} years = {LOOKBACK_DAYS} days)")
        print("   Results may not be statistically significant.")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease ensure .env file is properly configured.")
        return False
    
    return True


# Validate on import
if __name__ != "__main__":
    validate_config()