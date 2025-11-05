"""
10-Year Financial Sentiment Analysis Pipeline - RENDER VERSION
Main orchestrator with Flask API for web deployment
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
from flask import Flask, request, jsonify
from flask_cors import CORS

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
    """Generate statistical warnings about correlation strength and reliability."""
    warnings = []
    
    # Extract key metrics
    correlation = safe_get(correlation_results, 'price_sentiment_correlation', 0.0)
    p_value = safe_get(correlation_results, 'correlation_pvalue', 1.0)
    total_days = safe_get(correlation_results, 'total_days', 0)
    
    # Warning 1: Weak correlation
    if abs(correlation) < 0.3 and p_value < 0.05:
        warnings.append({
            'type': 'WEAK_CORRELATION',
            'severity': 'MEDIUM',
            'message': f'⚠️ Correlation is statistically significant (p={p_value:.4f}) but WEAK in magnitude ({correlation:+.3f}).'
        })
    
    # Warning 2: Very weak correlation
    if abs(correlation) < 0.15:
        warnings.append({
            'type': 'VERY_WEAK_CORRELATION',
            'severity': 'HIGH',
            'message': f'⚠️ VERY WEAK correlation ({correlation:+.3f}). Sentiment has minimal predictive power.'
        })
    
    # Warning 3: Not statistically significant
    if p_value >= 0.05:
        warnings.append({
            'type': 'NOT_SIGNIFICANT',
            'severity': 'HIGH',
            'message': f'⚠️ Correlation is NOT statistically significant (p={p_value:.4f} >= 0.05).'
        })
    
    # Warning 4: Insufficient data
    if total_days < 100:
        warnings.append({
            'type': 'INSUFFICIENT_DATA',
            'severity': 'HIGH',
            'message': f'⚠️ Only {total_days} days of data available.'
        })
    
    # Warning 5: Momentum
    momentum = safe_get(correlation_results, 'sentiment_momentum', 0.0)
    if momentum < -0.05:
        warnings.append({
            'type': 'NEGATIVE_MOMENTUM',
            'severity': 'INFO',
            'message': f'ℹ️ Negative sentiment momentum ({momentum:+.4f}) - sentiment declining over time.'
        })
    elif momentum > 0.05:
        warnings.append({
            'type': 'POSITIVE_MOMENTUM',
            'severity': 'INFO',
            'message': f'ℹ️ Positive sentiment momentum ({momentum:+.4f}) - sentiment improving over time.'
        })
    
    return warnings

def classify_sentiment(correlation_results):
    """Classify overall sentiment with robust NaN handling."""
    # Extract metrics with safe defaults
    sentiment_score = safe_get(correlation_results, 'avg_sentiment', 0.0)
    momentum = safe_get(correlation_results, 'sentiment_momentum', 0.0)
    correlation = safe_get(correlation_results, 'price_sentiment_correlation', 0.0)
    p_value = safe_get(correlation_results, 'correlation_pvalue', 1.0)
    
    # Weighted scoring
    final_score = (sentiment_score * 0.5) + (momentum * 0.3) + (correlation * 0.2)
    
    # Classification
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
    
    # Reliability
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

def analyze_ticker(ticker):
    """
    Run complete analysis pipeline with robust error handling.
    Returns dictionary with analysis results.
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
            return {'error': 'No price data found'}
        
        logger.info(f"✓ Fetched {len(price_df)} price records")
        logger.info(f"✓ Fetched {len(news_df)} news articles")
        logger.info(f"✓ Fetched {len(social_df)} social posts")
        
        # Step 2: Sentiment Analysis
        logger.info(f"\nStep 2/4: Analyzing sentiment for {ticker}...")
        sentiment_df = analyze_sentiment(news_df, social_df)
        
        if sentiment_df.empty:
            logger.warning(f"No sentiment data generated for {ticker}")
            return {'error': 'No sentiment data generated'}
        
        logger.info(f"✓ Generated sentiment scores for {len(sentiment_df)} days")
        
        # Validate sentiment data
        if 'combined_sentiment' not in sentiment_df.columns:
            logger.error("Missing combined_sentiment column")
            return {'error': 'Missing sentiment column'}
        
        # Clean NaN values
        nan_count = sentiment_df['combined_sentiment'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, cleaning...")
            sentiment_df['combined_sentiment'] = sentiment_df['combined_sentiment'].fillna(0.0)
        
        # Step 3: Price-Sentiment Correlation
        logger.info(f"\nStep 3/4: Correlating price and sentiment for {ticker}...")
        correlation_results = correlate_price_sentiment(price_df, sentiment_df)
        
        if 'error' in correlation_results:
            logger.error(f"Correlation failed: {correlation_results['error']}")
            return {'error': correlation_results['error']}
        
        # Step 4: Generate Warnings
        logger.info(f"\nStep 4/4: Generating statistical warnings for {ticker}...")
        warnings = generate_statistical_warnings(correlation_results)
        
        # Final Classification
        classification, confidence_score, reliability = classify_sentiment(correlation_results)
        
        # Compile results
        results = {
            'ticker': ticker,
            'company_name': company_name,
            'analysis_date': datetime.now().isoformat(),
            'lookback_years': float(config.LOOKBACK_YEARS),
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
            'statistical_warnings': warnings
        }
        
        # Convert to serializable
        results = convert_to_serializable(results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYSIS COMPLETE FOR {ticker}")
        logger.info(f"Classification: {classification}")
        logger.info(f"Confidence: {confidence_score:.4f}")
        logger.info(f"{'='*60}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}

# =============================================================================
# FLASK WEB SERVER (for Render deployment)
# =============================================================================

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Sentiment Analysis API is live! 🚀',
        'version': '2.0',
        'endpoints': {
            'analyze': 'POST /api/analyze with {"ticker": "AAPL"}',
            'health': 'GET / for health check'
        },
        'config': {
            'lookback_years': config.LOOKBACK_YEARS,
            'lookback_days': config.LOOKBACK_DAYS
        }
    })

@app.route('/api/analyze', methods=['POST', 'GET'])
def analyze_api():
    """Main analysis endpoint"""
    try:
        # Get ticker from request
        if request.method == 'GET':
            ticker = request.args.get('ticker', '').upper().strip()
        else:
            data = request.get_json() or {}
            ticker = data.get('ticker', '').upper().strip()
        
        if not ticker:
            return jsonify({
                'error': 'Ticker symbol required',
                'usage': 'POST /api/analyze with {"ticker": "AAPL"} or GET /api/analyze?ticker=AAPL'
            }), 400
        
        logger.info(f"==> API request received for ticker: {ticker}")
        
        # Run analysis
        result = analyze_ticker(ticker)
        
        # Check for errors
        if 'error' in result:
            return jsonify({
                'success': False,
                'ticker': ticker,
                'error': result['error']
            }), 500
        
        # Return success
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Analysis failed. Check server logs.'
        }), 500

# CRITICAL: Must be at the very end
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    print("="*70)
    print(f"🚀 Starting Flask server for Render deployment")
    print(f"🌐 Binding to 0.0.0.0:{port}")
    print(f"📊 Lookback period: {config.LOOKBACK_YEARS} years")
    print("="*70)
    
    app.run(
        host='0.0.0.0',  # MUST be 0.0.0.0 for Render
        port=port,
        debug=False,
        threaded=True
    )
