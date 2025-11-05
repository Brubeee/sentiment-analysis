"""
Price-Sentiment Correlation module - FIXED VERSION
Simple, robust analysis with proper NaN handling for 10-year investment decisions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
import config

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """
    Safely convert value to float, handling NaN/inf/None.
    Returns default value if conversion fails.
    """
    try:
        if value is None or pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def merge_price_sentiment(price_df, sentiment_df):
    """Merge price and sentiment data on date."""
    logger.info("Merging price and sentiment data...")
    
    price_df = price_df.copy()
    sentiment_df = sentiment_df.copy()
    
    # Normalize date columns
    if 'Date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['Date'])
    else:
        price_df['date'] = pd.to_datetime(price_df['date'])
    
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Remove timezone info
    if price_df['date'].dt.tz is not None:
        price_df['date'] = price_df['date'].dt.tz_localize(None)
    if sentiment_df['date'].dt.tz is not None:
        sentiment_df['date'] = sentiment_df['date'].dt.tz_localize(None)
    
    # Normalize to date only
    price_df['date'] = price_df['date'].dt.normalize()
    sentiment_df['date'] = sentiment_df['date'].dt.normalize()
    
    # Merge
    merged_df = price_df.merge(
        sentiment_df,
        on='date',
        how='inner',
        suffixes=('', '_sentiment')
    )
    
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Merged {len(merged_df)} days of data")
    logger.info(f"Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
    
    return merged_df


def calculate_simple_metrics(merged_df):
    """
    FIXED: Calculate simple metrics with robust NaN handling.
    All complex calculations removed, focus on core investment metrics.
    """
    logger.info("Calculating correlation metrics...")
    
    # Calculate daily returns
    merged_df['daily_return'] = merged_df['Close'].pct_change()
    
    # CRITICAL FIX: Drop rows with NaN in ANY relevant column
    valid_data = merged_df.dropna(subset=['daily_return', 'combined_sentiment', 'Close'])
    
    # Additional safety: Remove inf values
    valid_data = valid_data[~np.isinf(valid_data['daily_return'])]
    valid_data = valid_data[~np.isinf(valid_data['combined_sentiment'])]
    
    if len(valid_data) < 10:  # Need at least 10 days for meaningful analysis
        logger.warning("Insufficient data for analysis")
        return {
            'price_sentiment_correlation': 0.0,
            'correlation_pvalue': 1.0,
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'sentiment_momentum': 0.0,
            'positive_sentiment_days': 0,
            'negative_sentiment_days': 0,
            'neutral_sentiment_days': 0,
            'total_days': 0,
            'avg_return_positive_sentiment': 0.0,
            'avg_return_negative_sentiment': 0.0,
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0
        }
    
    # CRITICAL FIX: Explicit NaN handling for correlation
    # Remove any rows where either value is NaN
    corr_data = valid_data[['combined_sentiment', 'daily_return']].dropna()
    
    if len(corr_data) < 10:
        correlation, p_value = 0.0, 1.0
    else:
        try:
            correlation, p_value = stats.pearsonr(
                corr_data['combined_sentiment'],
                corr_data['daily_return']
            )
            # CRITICAL: Check if result is NaN
            if pd.isna(correlation) or np.isinf(correlation):
                correlation = 0.0
                p_value = 1.0
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            correlation, p_value = 0.0, 1.0
    
    # CRITICAL FIX: Safe sentiment statistics with explicit NaN handling
    sentiment_values = valid_data['combined_sentiment'].dropna()
    
    if len(sentiment_values) == 0:
        avg_sentiment = 0.0
        sentiment_std = 0.0
    else:
        avg_sentiment = safe_float(sentiment_values.mean(), 0.0)
        sentiment_std = safe_float(sentiment_values.std(), 0.0)
    
    # Sentiment distribution (simplified)
    positive_days = int((valid_data['combined_sentiment'] > 0.05).sum())
    negative_days = int((valid_data['combined_sentiment'] < -0.05).sum())
    neutral_days = int(len(valid_data) - positive_days - negative_days)
    
    # Simple momentum calculation (avoid overcomplicated rolling windows)
    # Just compare first half vs second half of data
    half_point = len(valid_data) // 2
    if half_point > 5:  # Need at least 5 days in each half
        first_half_sentiment = safe_float(valid_data['combined_sentiment'].iloc[:half_point].mean(), 0.0)
        second_half_sentiment = safe_float(valid_data['combined_sentiment'].iloc[half_point:].mean(), 0.0)
        sentiment_momentum = second_half_sentiment - first_half_sentiment
    else:
        sentiment_momentum = 0.0
    
    # Performance by sentiment (simplified)
    positive_sentiment_data = valid_data[valid_data['combined_sentiment'] > 0.05]
    negative_sentiment_data = valid_data[valid_data['combined_sentiment'] < -0.05]
    
    if len(positive_sentiment_data) > 0:
        avg_return_positive = safe_float(positive_sentiment_data['daily_return'].mean(), 0.0)
    else:
        avg_return_positive = 0.0
    
    if len(negative_sentiment_data) > 0:
        avg_return_negative = safe_float(negative_sentiment_data['daily_return'].mean(), 0.0)
    else:
        avg_return_negative = 0.0
    
    # FIXED: Price performance using actual date range (not just merged data points)
    if len(valid_data) > 1:
        first_price = safe_float(valid_data['Close'].iloc[0], 1.0)
        last_price = safe_float(valid_data['Close'].iloc[-1], 1.0)
        
        if first_price > 0:
            total_return = ((last_price / first_price) - 1) * 100
        else:
            total_return = 0.0
        
        # CRITICAL FIX: Calculate actual years from dates, not data points
        # Using data points (len/252) is wrong when we have gaps in sentiment data
        start_date = valid_data['date'].iloc[0]
        end_date = valid_data['date'].iloc[-1]
        actual_days = (end_date - start_date).days
        actual_years = actual_days / 365.25
        
        # Annualize return based on actual time period
        if actual_years > 0:
            # Use compound annual growth rate (CAGR) formula instead of simple division
            # CAGR = (ending_value / beginning_value)^(1/years) - 1
            annualized_return = (((last_price / first_price) ** (1 / actual_years)) - 1) * 100
        else:
            annualized_return = 0.0
    else:
        total_return = 0.0
        annualized_return = 0.0
    
    # CRITICAL: Ensure all values are safe floats
    metrics = {
        'price_sentiment_correlation': safe_float(correlation, 0.0),
        'correlation_pvalue': safe_float(p_value, 1.0),
        'avg_sentiment': safe_float(avg_sentiment, 0.0),
        'sentiment_std': safe_float(sentiment_std, 0.0),
        'sentiment_momentum': safe_float(sentiment_momentum, 0.0),
        'positive_sentiment_days': int(positive_days),
        'negative_sentiment_days': int(negative_days),
        'neutral_sentiment_days': int(neutral_days),
        'total_days': int(len(valid_data)),
        'avg_return_positive_sentiment': safe_float(avg_return_positive, 0.0),
        'avg_return_negative_sentiment': safe_float(avg_return_negative, 0.0),
        'total_return_pct': safe_float(total_return, 0.0),
        'annualized_return_pct': safe_float(annualized_return, 0.0)
    }
    
    return metrics


def analyze_sentiment_periods(merged_df):
    """Simple breakdown: price performance during different sentiment periods."""
    logger.info("Analyzing sentiment periods...")
    
    if 'daily_return' not in merged_df.columns:
        merged_df['daily_return'] = merged_df['Close'].pct_change()
    
    valid_data = merged_df.dropna(subset=['daily_return', 'combined_sentiment'])
    
    if len(valid_data) == 0:
        return {
            'Negative': {'count': 0, 'avg_daily_return': 0.0, 'win_rate': 0.0},
            'Neutral': {'count': 0, 'avg_daily_return': 0.0, 'win_rate': 0.0},
            'Positive': {'count': 0, 'avg_daily_return': 0.0, 'win_rate': 0.0}
        }
    
    # Define simple categories
    valid_data['sentiment_category'] = pd.cut(
        valid_data['combined_sentiment'],
        bins=[-np.inf, -0.1, 0.1, np.inf],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    performance = {}
    
    for category in ['Negative', 'Neutral', 'Positive']:
        cat_data = valid_data[valid_data['sentiment_category'] == category]
        
        if len(cat_data) > 0:
            avg_return = safe_float(cat_data['daily_return'].mean(), 0.0)
            win_count = (cat_data['daily_return'] > 0).sum()
            win_rate = safe_float((win_count / len(cat_data) * 100), 0.0)
            
            performance[category] = {
                'count': int(len(cat_data)),
                'avg_daily_return': avg_return,
                'win_rate': win_rate
            }
        else:
            performance[category] = {
                'count': 0,
                'avg_daily_return': 0.0,
                'win_rate': 0.0
            }
    
    return performance


def correlate_price_sentiment(price_df, sentiment_df):
    """
    FIXED: Main correlation analysis function with robust error handling.
    """
    logger.info("\n" + "="*60)
    logger.info("PRICE-SENTIMENT CORRELATION ANALYSIS")
    logger.info("="*60 + "\n")
    
    if price_df.empty or sentiment_df.empty:
        logger.error("Empty input data")
        return {
            'error': 'Empty input data',
            'price_sentiment_correlation': 0.0,
            'correlation_pvalue': 1.0,
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'sentiment_momentum': 0.0,
            'positive_sentiment_days': 0,
            'negative_sentiment_days': 0,
            'neutral_sentiment_days': 0,
            'total_days': 0,
            'avg_return_positive_sentiment': 0.0,
            'avg_return_negative_sentiment': 0.0,
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0
        }
    
    try:
        # Merge data
        merged_df = merge_price_sentiment(price_df, sentiment_df)
        
        if merged_df.empty:
            logger.error("No overlapping data")
            return {
                'error': 'No overlapping data',
                'price_sentiment_correlation': 0.0,
                'correlation_pvalue': 1.0,
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_momentum': 0.0
            }
        
        # Calculate metrics
        metrics = calculate_simple_metrics(merged_df)
        
        # Analyze sentiment periods
        period_performance = analyze_sentiment_periods(merged_df)
        
        # Combine results
        results = {
            **metrics,
            'sentiment_period_performance': period_performance,
            'date_range': {
                'start': str(merged_df['date'].min().date()),
                'end': str(merged_df['date'].max().date())
            }
        }
        
        # Log summary
        logger.info("RESULTS:")
        logger.info("-" * 60)
        logger.info(f"Correlation: {metrics['price_sentiment_correlation']:.3f} (p={metrics['correlation_pvalue']:.3f})")
        logger.info(f"Average Sentiment: {metrics['avg_sentiment']:.3f}")
        logger.info(f"Sentiment Momentum: {metrics['sentiment_momentum']:.3f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.1f}%")
        logger.info(f"Annualized Return: {metrics['annualized_return_pct']:.1f}%")
        logger.info("")
        logger.info(f"Positive Sentiment Days: {metrics['positive_sentiment_days']} ({metrics['positive_sentiment_days']/max(metrics['total_days'], 1)*100:.1f}%)")
        logger.info(f"  → Avg Daily Return: {metrics['avg_return_positive_sentiment']:.3%}")
        logger.info(f"Negative Sentiment Days: {metrics['negative_sentiment_days']} ({metrics['negative_sentiment_days']/max(metrics['total_days'], 1)*100:.1f}%)")
        logger.info(f"  → Avg Daily Return: {metrics['avg_return_negative_sentiment']:.3%}")
        logger.info("")
        logger.info("Sentiment Period Performance:")
        for period, perf in period_performance.items():
            logger.info(f"  {period:8s}: {perf['count']:3d} days, Avg Return: {perf['avg_daily_return']:+.3%}, Win Rate: {perf['win_rate']:.1f}%")
        logger.info("="*60 + "\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'price_sentiment_correlation': 0.0,
            'correlation_pvalue': 1.0,
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'sentiment_momentum': 0.0,
            'positive_sentiment_days': 0,
            'negative_sentiment_days': 0,
            'neutral_sentiment_days': 0,
            'total_days': 0,
            'avg_return_positive_sentiment': 0.0,
            'avg_return_negative_sentiment': 0.0,
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0
        }