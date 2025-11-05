# api/analyze.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST', 'GET'])
def analyze():
    """
    Lightweight sentiment analysis using external APIs
    """
    # Get ticker from request
    if request.method == 'GET':
        ticker = request.args.get('ticker', '').upper().strip()
    else:
        data = request.get_json() or {}
        ticker = data.get('ticker', '').upper().strip()
    
    if not ticker:
        return jsonify({
            'error': 'Ticker symbol required',
            'usage': 'POST /api/analyze with {"ticker": "AAPL"}'
        }), 400
    
    try:
        print(f"Analyzing {ticker}...")
        
        # Step 1: Get price data (fast - yfinance is small)
        price_data = get_price_data(ticker)
        
        # Step 2: Get news headlines (fast)
        news_headlines = get_news_data(ticker)
        
        # Step 3: Analyze sentiment using Hugging Face API (no local model!)
        sentiment_result = analyze_sentiment_hf(news_headlines)
        
        # Step 4: Calculate final score
        classification = classify_sentiment(sentiment_result['score'])
        
        result = {
            'success': True,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'classification': classification,
                'sentiment_score': round(sentiment_result['score'], 4),
                'confidence': round(sentiment_result['confidence'], 4),
                'current_price': price_data['current_price'],
                'price_change_pct': price_data['change_pct'],
                'news_count': len(news_headlines),
                'message': f'Analysis complete for {ticker}'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'ticker': ticker
        }), 500

def get_price_data(ticker):
    """Fetch latest price data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')
        
        if hist.empty:
            raise ValueError(f"No price data found for {ticker}")
        
        current_price = float(hist['Close'].iloc[-1])
        start_price = float(hist['Close'].iloc[0])
        change_pct = ((current_price / start_price) - 1) * 100
        
        return {
            'current_price': round(current_price, 2),
            'change_pct': round(change_pct, 2)
        }
    except Exception as e:
        print(f"Price data error: {e}")
        return {'current_price': 0, 'change_pct': 0}

def get_news_data(ticker):
    """Fetch news headlines from Finnhub API"""
    try:
        api_key = os.environ.get('FINNHUB_API_KEY', '')
        if not api_key:
            print("Warning: No Finnhub API key found")
            return [f"Sample news about {ticker}"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days only
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            articles = response.json()
            headlines = [article.get('headline', '') for article in articles[:20]]
            return [h for h in headlines if h]  # Remove empty
        else:
            print(f"Finnhub API error: {response.status_code}")
            return [f"Sample news about {ticker}"]
            
    except Exception as e:
        print(f"News fetch error: {e}")
        return [f"Sample news about {ticker}"]

def analyze_sentiment_hf(headlines):
    """
    Analyze sentiment using Hugging Face Inference API
    NO MODEL DOWNLOAD - uses HF's servers!
    """
    try:
        api_key = os.environ.get('HUGGINGFACE_API_KEY', '')
        if not api_key:
            print("Warning: No Hugging Face API key - using fallback")
            return {'score': 0.0, 'confidence': 0.5}
        
        # Combine headlines (limit text length)
        combined_text = ". ".join(headlines[:10])[:500]
        
        # Call Hugging Face FinBERT model
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": combined_text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse HF response format: [[{'label': 'positive', 'score': 0.95}, ...]]
            if isinstance(result, list) and len(result) > 0:
                sentiments = result[0]
                
                # Extract positive and negative scores
                pos_score = 0
                neg_score = 0
                
                for item in sentiments:
                    if item['label'].lower() == 'positive':
                        pos_score = item['score']
                    elif item['label'].lower() == 'negative':
                        neg_score = item['score']
                
                # Net sentiment score (-1 to +1)
                net_score = pos_score - neg_score
                confidence = max(pos_score, neg_score)
                
                return {
                    'score': net_score,
                    'confidence': confidence
                }
        
        # Fallback if API call fails
        print(f"HF API error: {response.status_code}")
        return {'score': 0.0, 'confidence': 0.5}
        
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return {'score': 0.0, 'confidence': 0.5}

def classify_sentiment(score):
    """Classify sentiment score into categories"""
    if score >= 0.7:
        return 'SUPER_POSITIVE'
    elif score >= 0.25:
        return 'POSITIVE'
    elif score >= -0.25:
        return 'NEUTRAL'
    elif score >= -0.7:
        return 'NEGATIVE'
    else:
        return 'SUPER_NEGATIVE'

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running',
        'timestamp': datetime.now().isoformat()
    })

# For local testing only
if __name__ == '__main__':
    app.run(debug=True, port=5000)
