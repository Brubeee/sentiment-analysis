"""
10-Year Financial Sentiment Analysis Pipeline - FIXED VERSION
Main orchestrator with robust NaN handling for investment decisions.
UPDATED: Includes statistical warnings and clearer interpretation
"""

# CRITICAL: Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import config
from data_fetching import fetch_all_data
from sentiment_analysis import analyze_sentiment
from price_correlation import correlate_price_sentiment
import json
import os
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_get(dictionary, key, default=0.0):
    """Safely get value from dictionary with NaN handling."""
    value = dictionary.get(key, default)
    if pd.isna(value) or (isinstance(value, (int, float)) and np.isinf(value)):
        return default
    return value


def generate_statistical_warnings(correlation_results):
    """
    Generate statistical warnings about correlation strength and reliability.
    
    Returns:
        list: Warning messages to display to user
    """
    warnings = []
    
    # Extract key metrics
    correlation = safe_get(correlation_results, 'price_sentiment_correlation', 0.0)
    p_value = safe_get(correlation_results, 'correlation_pvalue', 1.0)
    total_days = safe_get(correlation_results, 'total_days', 0)
    
    # Warning 1: Weak correlation (even if significant)
    if abs(correlation) < 0.3 and p_value < 0.05:
        warnings.append({
            'type': 'WEAK_CORRELATION',
            'severity': 'MEDIUM',
            'message': f'⚠️  Correlation is statistically significant (p={p_value:.4f}) but WEAK in magnitude ({correlation:+.3f}). Sentiment explains only ~{(correlation**2)*100:.1f}% of price variance. Do not over-rely on sentiment alone.'
        })
    
    # Warning 2: Very weak correlation
    if abs(correlation) < 0.15:
        warnings.append({
            'type': 'VERY_WEAK_CORRELATION',
            'severity': 'HIGH',
            'message': f'⚠️  VERY WEAK correlation ({correlation:+.3f}). Sentiment has minimal predictive power for this stock. Other factors likely dominate price movements.'
        })
    
    # Warning 3: Not statistically significant
    if p_value >= 0.05:
        warnings.append({
            'type': 'NOT_SIGNIFICANT',
            'severity': 'HIGH',
            'message': f'⚠️  Correlation is NOT statistically significant (p={p_value:.4f} >= 0.05). Results may be due to random chance. Interpret with extreme caution.'
        })
    
    # Warning 4: Insufficient data
    if total_days < 100:
        warnings.append({
            'type': 'INSUFFICIENT_DATA',
            'severity': 'HIGH',
            'message': f'⚠️  Only {total_days} days of data available. Increase LOOKBACK_YEARS for more reliable results. Minimum 252 days (1 year) recommended.'
        })
    
    # Warning 5: Negative momentum explanation
    momentum = safe_get(correlation_results, 'sentiment_momentum', 0.0)
    if momentum < -0.05:
        warnings.append({
            'type': 'NEGATIVE_MOMENTUM',
            'severity': 'INFO',
            'message': f'ℹ️  Negative sentiment momentum ({momentum:+.4f}) means sentiment has been DECLINING over time (second half worse than first half). This suggests worsening public perception.'
        })
    elif momentum > 0.05:
        warnings.append({
            'type': 'POSITIVE_MOMENTUM',
            'severity': 'INFO',
            'message': f'ℹ️  Positive sentiment momentum ({momentum:+.4f}) means sentiment has been IMPROVING over time (second half better than first half). This suggests strengthening public perception.'
        })
    
    # Warning 6: Model limitations reminder
    warnings.append({
        'type': 'MODEL_LIMITATIONS',
        'severity': 'INFO',
        'message': '📊 REMINDER: Sentiment is ONE indicator among many. Consider fundamentals, technicals, market conditions, and macroeconomics. Past sentiment does NOT guarantee future returns.'
    })
    
    return warnings


def classify_sentiment(correlation_results):
    """
    FIXED: Classify overall sentiment with robust NaN handling.
    Simplified for long-term portfolio investment decisions.
    
    Focus on three core metrics:
    1. Average sentiment (most important - 50%)
    2. Sentiment momentum (trend direction - 30%)
    3. Price correlation (sentiment-price relationship - 20%)
    
    Returns: 
        tuple: (classification string, confidence score, reliability_note)
    """
    # Extract metrics with safe defaults
    sentiment_score = safe_get(correlation_results, 'avg_sentiment', 0.0)
    momentum = safe_get(correlation_results, 'sentiment_momentum', 0.0)
    correlation = safe_get(correlation_results, 'price_sentiment_correlation', 0.0)
    p_value = safe_get(correlation_results, 'correlation_pvalue', 1.0)
    
    # Weighted scoring for long-term outlook
    final_score = (sentiment_score * 0.5) + (momentum * 0.3) + (correlation * 0.2)
    
    # Classification thresholds
    if final_score > 0.6:
        classification = "SUPER POSITIVE"
    elif final_score > 0.2:
        classification = "POSITIVE"
    elif final_score > -0.2:
        classification = "NEUTRAL"
    elif final_score > -0.6:
        classification = "NEGATIVE"
    else:
        classification = "SUPER NEGATIVE"
    
    # Add reliability assessment
    if p_value >= 0.05:
        reliability = "LOW - Not statistically significant"
    elif abs(correlation) < 0.15:
        reliability = "LOW - Very weak correlation"
    elif abs(correlation) < 0.3:
        reliability = "MODERATE - Weak but significant correlation"
    elif abs(correlation) < 0.5:
        reliability = "GOOD - Moderate correlation"
    else:
        reliability = "HIGH - Strong correlation"
    
    return classification, final_score, reliability


def save_results(ticker, results, output_dir='results'):
    """Save comprehensive analysis results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{ticker}_analysis_{timestamp}.json"
    
    # CRITICAL: Convert any remaining numpy types to Python types
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    results = convert_to_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filename}")
    return filename


def print_warnings(warnings):
    """Print warnings in a formatted, easy-to-read way."""
    if not warnings:
        return
    
    logger.info("\n" + "🚨 " + "="*58)
    logger.info("STATISTICAL WARNINGS & INTERPRETATION GUIDANCE")
    logger.info("="*60)
    
    for i, warning in enumerate(warnings, 1):
        severity = warning['severity']
        message = warning['message']
        
        # Color coding by severity (for visual emphasis in logs)
        if severity == 'HIGH':
            prefix = "❗ IMPORTANT"
        elif severity == 'MEDIUM':
            prefix = "⚠️  CAUTION"
        else:
            prefix = "ℹ️  NOTE"
        
        logger.info(f"\n{prefix} ({i}/{len(warnings)}):")
        logger.info(f"{message}")
    
    logger.info("\n" + "="*60)
    logger.info("⚠️  READ WARNINGS CAREFULLY BEFORE MAKING DECISIONS")
    logger.info("="*60 + "\n")


def analyze_ticker(ticker):
    """
    FIXED: Run complete analysis pipeline with robust error handling.
    UPDATED: Includes statistical warnings and interpretation guidance.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting analysis for {ticker}")
    logger.info(f"Period: {config.LOOKBACK_YEARS} years ({config.LOOKBACK_DAYS} days)")
    logger.info(f"{'='*60}\n")
    
    try:
        # Step 1: Data Collection
        logger.info(f"Step 1/4: Fetching data for {ticker}...")
        price_df, news_df, social_df, company_name = fetch_all_data(ticker)
        
        if price_df.empty:
            logger.error(f"No price data found for {ticker}")
            return None
        
        logger.info(f"✓ Fetched {len(price_df)} price records")
        logger.info(f"✓ Fetched {len(news_df)} news articles")
        logger.info(f"✓ Fetched {len(social_df)} social posts")
        
        # Step 2: Sentiment Analysis
        logger.info(f"\nStep 2/4: Analyzing sentiment for {ticker}...")
        sentiment_df = analyze_sentiment(news_df, social_df)
        
        if sentiment_df.empty:
            logger.warning(f"No sentiment data generated for {ticker}")
            return None
        
        logger.info(f"✓ Generated sentiment scores for {len(sentiment_df)} days")
        
        # CRITICAL: Validate sentiment data before correlation
        if 'combined_sentiment' not in sentiment_df.columns:
            logger.error("Missing combined_sentiment column")
            return None
        
        nan_count = sentiment_df['combined_sentiment'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in sentiment data, cleaning...")
            sentiment_df['combined_sentiment'] = sentiment_df['combined_sentiment'].fillna(0.0)
        
        # Step 3: Price-Sentiment Correlation
        logger.info(f"\nStep 3/4: Correlating price and sentiment for {ticker}...")
        correlation_results = correlate_price_sentiment(price_df, sentiment_df)
        
        if 'error' in correlation_results:
            logger.error(f"Correlation analysis failed: {correlation_results['error']}")
            return None
        
        # Step 4: Generate Warnings (NEW)
        logger.info(f"\nStep 4/5: Generating statistical warnings for {ticker}...")
        warnings = generate_statistical_warnings(correlation_results)
        
        # Step 5: Final Classification
        logger.info(f"\nStep 5/5: Generating final classification for {ticker}...")
        classification, confidence_score, reliability = classify_sentiment(correlation_results)
        
        # Compile comprehensive results
        results = {
            'ticker': ticker,
            'company_name': company_name,
            'analysis_date': datetime.now().isoformat(),
            'lookback_years': float(config.LOOKBACK_YEARS),
            'lookback_days': int(config.LOOKBACK_DAYS),
            'data_summary': {
                'price_records': int(len(price_df)),
                'news_articles': int(len(news_df)),
                'social_posts': int(len(social_df)),
                'sentiment_days': int(len(sentiment_df))
            },
            'sentiment_metrics': correlation_results,
            'final_classification': classification,
            'confidence_score': float(confidence_score),
            'reliability_assessment': reliability,
            'statistical_warnings': warnings  # NEW: Include warnings in output
        }
        
        # Save results
        output_file = save_results(ticker, results)
        
        # Print warnings BEFORE summary (NEW)
        print_warnings(warnings)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYSIS COMPLETE FOR {ticker}")
        logger.info(f"{'='*60}")
        logger.info(f"Company: {company_name}")
        logger.info(f"Period: {config.LOOKBACK_YEARS} years")
        logger.info(f"")
        logger.info(f"LONG-TERM SENTIMENT: {classification}")
        logger.info(f"Confidence Score: {confidence_score:.4f}")
        logger.info(f"Reliability: {reliability}")
        logger.info(f"")
        logger.info(f"Key Metrics:")
        logger.info(f"  Average Sentiment: {safe_get(correlation_results, 'avg_sentiment', 0):+.4f}")
        logger.info(f"  Sentiment Momentum: {safe_get(correlation_results, 'sentiment_momentum', 0):+.4f}")
        
        # Add interpretation for momentum
        momentum = safe_get(correlation_results, 'sentiment_momentum', 0)
        if momentum < 0:
            logger.info(f"    → Sentiment DECLINING over time (worsening)")
        else:
            logger.info(f"    → Sentiment IMPROVING over time (strengthening)")
        
        logger.info(f"  Price Correlation: {safe_get(correlation_results, 'price_sentiment_correlation', 0):+.4f}")
        logger.info(f"    (p-value: {safe_get(correlation_results, 'correlation_pvalue', 1):.4f})")
        
        # Interpret correlation strength
        corr = abs(safe_get(correlation_results, 'price_sentiment_correlation', 0))
        if corr < 0.15:
            logger.info(f"    → VERY WEAK - Sentiment barely predicts price")
        elif corr < 0.3:
            logger.info(f"    → WEAK - Limited predictive power")
        elif corr < 0.5:
            logger.info(f"    → MODERATE - Some predictive power")
        else:
            logger.info(f"    → STRONG - Good predictive power")
        
        logger.info(f"  Total Return: {safe_get(correlation_results, 'total_return_pct', 0):+.2f}%")
        logger.info(f"  Annualized Return: {safe_get(correlation_results, 'annualized_return_pct', 0):+.2f}%")
        logger.info(f"")
        logger.info(f"Results saved: {output_file}")
        logger.info(f"{'='*60}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main entry point for the sentiment analysis pipeline."""
    logger.info("="*60)
    logger.info("Financial Sentiment Analysis Pipeline")
    logger.info(f"Lookback Period: {config.LOOKBACK_YEARS} years ({config.LOOKBACK_DAYS} days)")
    logger.info("="*60)
    
    # Define tickers to analyze
    tickers = ['LLY']
    
    # You can add multiple tickers:
    # tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    logger.info(f"Tickers to analyze: {', '.join(tickers)}")
    
    all_results = {}
    successful = 0
    failed = 0
    
    for ticker in tickers:
        result = analyze_ticker(ticker)
        if result:
            all_results[ticker] = result
            successful += 1
        else:
            failed += 1
    
    # Save combined summary if we have results
    if all_results:
        summary_file = save_results('SUMMARY', {
            'analysis_date': datetime.now().isoformat(),
            'lookback_years': float(config.LOOKBACK_YEARS),
            'lookback_days': int(config.LOOKBACK_DAYS),
            'tickers_analyzed': len(all_results),
            'successful': successful,
            'failed': failed,
            'results': all_results
        })
        logger.info(f"\nCombined summary saved to {summary_file}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Successful: {successful}/{len(tickers)}")
    logger.info(f"Failed: {failed}/{len(tickers)}")
    logger.info("="*60)
    
    # Final disclaimer
    logger.info("\n⚠️  IMPORTANT DISCLAIMER:")
    logger.info("This analysis is for educational/research purposes only.")
    logger.info("Sentiment analysis has limitations - see warnings above.")
    logger.info("Always conduct thorough research before investment decisions.")
    logger.info("Past performance does not guarantee future results.\n")


if __name__ == "__main__":
    main()