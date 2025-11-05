"""
Data fetching module with INTERNATIONAL STOCK SUPPORT
Handles fetching from: yfinance (prices), GDELT (news), Reddit (social)

NEW: Supports Indian, European, Asian, and other international stocks
"""

import yfinance as yf
import praw
import pandas as pd
from datetime import datetime, timedelta
import logging
import config
import time
import requests

logger = logging.getLogger(__name__)


def parse_ticker_info(ticker):
    """
    Parse ticker to identify market and extract clean company identifier.
    
    Returns:
        dict with 'market', 'clean_ticker', 'search_terms'
    """
    ticker_upper = ticker.upper()
    
    # Indian NSE
    if ticker_upper.endswith('.NS'):
        return {
            'market': 'IN-NSE',
            'clean_ticker': ticker_upper.replace('.NS', ''),
            'search_terms': [ticker_upper.replace('.NS', '')],
            'subreddits': ['IndianStreetBets', 'IndiaSpeaks', 'IndianStockMarket', 'investing']
        }
    
    # Indian BSE
    elif ticker_upper.endswith('.BO'):
        return {
            'market': 'IN-BSE',
            'clean_ticker': ticker_upper.replace('.BO', ''),
            'search_terms': [ticker_upper.replace('.BO', '')],
            'subreddits': ['IndianStreetBets', 'IndiaSpeaks', 'IndianStockMarket', 'investing']
        }
    
    # London Stock Exchange
    elif ticker_upper.endswith('.L'):
        return {
            'market': 'UK-LSE',
            'clean_ticker': ticker_upper.replace('.L', ''),
            'search_terms': [ticker_upper.replace('.L', '')],
            'subreddits': ['UKInvesting', 'stocks', 'investing']
        }
    
    # Tokyo Stock Exchange
    elif ticker_upper.endswith('.T'):
        return {
            'market': 'JP-TSE',
            'clean_ticker': ticker_upper.replace('.T', ''),
            'search_terms': [ticker_upper.replace('.T', '')],
            'subreddits': ['JapanFinance', 'stocks', 'investing']
        }
    
    # Hong Kong Stock Exchange
    elif ticker_upper.endswith('.HK'):
        return {
            'market': 'HK-HKEX',
            'clean_ticker': ticker_upper.replace('.HK', ''),
            'search_terms': [ticker_upper.replace('.HK', '')],
            'subreddits': ['HongKong', 'stocks', 'investing']
        }
    
    # Toronto Stock Exchange
    elif ticker_upper.endswith('.TO'):
        return {
            'market': 'CA-TSX',
            'clean_ticker': ticker_upper.replace('.TO', ''),
            'search_terms': [ticker_upper.replace('.TO', '')],
            'subreddits': ['CanadianInvestor', 'stocks', 'investing']
        }
    
    # Australian Stock Exchange
    elif ticker_upper.endswith('.AX'):
        return {
            'market': 'AU-ASX',
            'clean_ticker': ticker_upper.replace('.AX', ''),
            'search_terms': [ticker_upper.replace('.AX', '')],
            'subreddits': ['AusFinance', 'ASX_Bets', 'stocks', 'investing']
        }
    
    # German Stock Exchange
    elif ticker_upper.endswith('.DE'):
        return {
            'market': 'DE-XETRA',
            'clean_ticker': ticker_upper.replace('.DE', ''),
            'search_terms': [ticker_upper.replace('.DE', '')],
            'subreddits': ['mauerstrassenwetten', 'Finanzen', 'stocks', 'investing']
        }
    
    # US stocks (default - no suffix)
    else:
        return {
            'market': 'US-NYSE/NASDAQ',
            'clean_ticker': ticker_upper,
            'search_terms': [ticker_upper, f'${ticker_upper}'],
            'subreddits': config.REDDIT_SUBREDDITS  # Use config default
        }


def fetch_price_data(ticker):
    """
    Fetch price history for ANY international ticker using yfinance.
    
    Returns:
        tuple: (price_df, company_name)
    """
    logger.info(f"Fetching price data for {ticker}...")
    
    try:
        # Parse ticker info
        ticker_info = parse_ticker_info(ticker)
        logger.info(f"Detected market: {ticker_info['market']}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch company name
        info = stock.info
        company_name = info.get('longName', info.get('shortName', ticker))
        logger.info(f"Company name: {company_name}")
        
        # Fetch price history
        price_df = stock.history(
            start=config.START_DATE,
            end=config.END_DATE,
            interval='1d'
        )
        
        if price_df.empty:
            logger.warning(f"No price data found for {ticker}")
            return pd.DataFrame(), ticker
        
        # Reset index to make Date a column
        price_df.reset_index(inplace=True)
        
        # Ensure date column is datetime
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        logger.info(f"Fetched {len(price_df)} price records from {price_df['Date'].min().date()} to {price_df['Date'].max().date()}")
        
        return price_df, company_name
        
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {str(e)}")
        return pd.DataFrame(), ticker


def fetch_social_data(ticker):
    """
    Fetch Reddit posts for the given ticker with INTERNATIONAL SUPPORT.
    Automatically detects market and uses appropriate subreddits.
    
    Returns:
        DataFrame with columns [date, text, score, subreddit]
    """
    logger.info(f"Fetching social media data for {ticker}...")
    
    # Check for API credentials
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        logger.warning("Reddit API credentials not found. Skipping social data...")
        return pd.DataFrame(columns=['date', 'text', 'score', 'subreddit'])
    
    try:
        # Parse ticker to get appropriate subreddits
        ticker_info = parse_ticker_info(ticker)
        search_terms = ticker_info['search_terms']
        subreddits = ticker_info['subreddits']
        
        logger.info(f"Using search terms: {search_terms}")
        logger.info(f"Target subreddits: {subreddits}")
        
        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )
        
        all_posts = []
        
        # Search across appropriate subreddits for the market
        for subreddit_name in subreddits:
            logger.info(f"Searching r/{subreddit_name} for {ticker}...")
            
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Search for posts mentioning the ticker
                for query in search_terms:
                    try:
                        # Search with time filter
                        for post in subreddit.search(
                            query,
                            limit=config.REDDIT_POST_LIMIT,
                            time_filter='all',
                            sort='relevance'
                        ):
                            # Filter by date range
                            post_date = datetime.fromtimestamp(post.created_utc)
                            if config.START_DATE <= post_date <= config.END_DATE:
                                all_posts.append({
                                    'date': post_date,
                                    'text': f"{post.title} {post.selftext}",
                                    'score': post.score,
                                    'subreddit': subreddit_name
                                })
                    except Exception as e:
                        logger.warning(f"Error searching {query} in r/{subreddit_name}: {str(e)}")
                        continue
                    
                    # Rate limiting
                    time.sleep(1)
                    
            except Exception as e:
                logger.warning(f"Error accessing r/{subreddit_name}: {str(e)}")
                continue
        
        if not all_posts:
            logger.warning(f"No social media posts found for {ticker}")
            return pd.DataFrame(columns=['date', 'text', 'score', 'subreddit'])
        
        social_df = pd.DataFrame(all_posts)
        social_df['date'] = pd.to_datetime(social_df['date'])
        social_df = social_df.sort_values('date')
        
        logger.info(f"Fetched {len(social_df)} social media posts")
        
        return social_df
        
    except Exception as e:
        logger.error(f"Error fetching social data: {str(e)}")
        return pd.DataFrame(columns=['date', 'text', 'score', 'subreddit'])


def fetch_news_data_gdelt(ticker, company_name):
    """
    Fetch news articles using GDELT with INTERNATIONAL SUPPORT.
    Uses clean company name for better international news coverage.
    
    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS", "AAPL")
        company_name: Full company name (e.g., "Reliance Industries Limited")
    
    Returns:
        DataFrame with columns [date, title, url, source]
    """
    logger.info(f"Fetching news data from GDELT for {ticker}...")
    logger.info(f"Time period: {config.LOOKBACK_YEARS} years ({config.LOOKBACK_DAYS} days)")
    
    # Parse ticker info
    ticker_info = parse_ticker_info(ticker)
    clean_ticker = ticker_info['clean_ticker']
    
    # Build smart query
    # For international stocks, use company name + clean ticker
    # For US stocks, use ticker variations
    if ticker_info['market'] == 'US-NYSE/NASDAQ':
        # US stocks: use ticker-focused search
        query_terms = [clean_ticker, f'${clean_ticker}']
        query = f'({" OR ".join(query_terms)}) (stock OR market OR shares OR trading)'
    else:
        # International stocks: use company name + ticker
        # Extract first part of company name (before "Limited", "Inc.", etc.)
        company_base = company_name.split(' Limited')[0].split(' Inc')[0].split(' Corporation')[0]
        query_terms = [company_base, clean_ticker]
        query = f'({" OR ".join(query_terms)}) (stock OR market OR shares)'
    
    logger.info(f"GDELT Query: {query}")
    
    all_articles = []
    
    try:
        # GDELT 2.0 DOC API endpoint
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Calculate number of time chunks
        if config.LOOKBACK_YEARS < 1.0:
            chunk_days = 30
            articles_per_chunk = max(50, int(config.TOTAL_ARTICLES_TARGET / max(1, config.LOOKBACK_DAYS / 30)))
        else:
            chunk_days = 365
            articles_per_chunk = config.ARTICLES_PER_YEAR
        
        num_chunks = max(1, int(config.LOOKBACK_DAYS / chunk_days))
        logger.info(f"Fetching in {num_chunks} chunks of ~{chunk_days} days each")
        logger.info(f"Target: ~{articles_per_chunk} articles per chunk")
        
        # Fetch data in chunks
        for chunk_idx in range(num_chunks):
            chunk_end = config.END_DATE - timedelta(days=chunk_idx * chunk_days)
            chunk_start = config.END_DATE - timedelta(days=(chunk_idx + 1) * chunk_days)
            
            if chunk_start < config.START_DATE:
                chunk_start = config.START_DATE
            
            logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: {chunk_start.date()} to {chunk_end.date()}")
            
            try:
                # Format dates for GDELT API
                start_str = chunk_start.strftime('%Y%m%d000000')
                end_str = chunk_end.strftime('%Y%m%d235959')
                
                # Build API request
                params = {
                    'query': query,
                    'mode': 'artlist',
                    'maxrecords': min(articles_per_chunk, 250),
                    'startdatetime': start_str,
                    'enddatetime': end_str,
                    'format': 'json',
                    'sort': 'datedesc'
                }
                
                # Make request with retry logic
                max_retries = 3
                response = None
                for attempt in range(max_retries):
                    try:
                        response = requests.get(base_url, params=params, timeout=30)
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"  -> Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                            time.sleep(2 ** attempt)
                        else:
                            logger.warning(f"  -> Request failed after {max_retries} attempts: {str(e)}")
                            response = None
                            break
                
                if response and response.status_code == 200:
                    if not response.text or response.text.strip() == '':
                        logger.warning(f"  -> Empty response from GDELT API")
                        continue
                    
                    try:
                        data = response.json()
                    except ValueError:
                        logger.warning(f"  -> Invalid JSON response")
                        continue
                    
                    if 'articles' in data and data['articles']:
                        articles = data['articles']
                        logger.info(f"  -> Received {len(articles)} articles")
                        
                        # Parse articles
                        for article in articles[:articles_per_chunk]:
                            try:
                                article_date_str = article.get('seendate', start_str)
                                article_date = pd.to_datetime(article_date_str, format='%Y%m%dT%H%M%SZ', errors='coerce')
                                
                                if pd.isna(article_date):
                                    article_date = chunk_start
                                
                                all_articles.append({
                                    'date': article_date,
                                    'title': article.get('title', '')[:500],
                                    'url': article.get('url', ''),
                                    'source': article.get('domain', 'GDELT'),
                                    'language': article.get('language', '')
                                })
                            except Exception as e:
                                logger.debug(f"Error parsing article: {str(e)}")
                                continue
                        
                        logger.info(f"  -> Collected {len(articles[:articles_per_chunk])} articles")
                    else:
                        logger.warning(f"  -> No articles found")
                else:
                    if response:
                        logger.warning(f"  -> API request failed with status {response.status_code}")
                
                # Rate limiting
                time.sleep(2)
                
                if chunk_start <= config.START_DATE:
                    break
                
            except Exception as e:
                logger.warning(f"Error fetching chunk {chunk_idx + 1}: {str(e)}")
                continue
        
        if not all_articles:
            logger.warning(f"No news articles found for {ticker}")
            return pd.DataFrame(columns=['date', 'title', 'url', 'source'])
        
        # Create DataFrame
        news_df = pd.DataFrame(all_articles)
        logger.info(f"Created DataFrame with {len(news_df)} raw articles")
        
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = news_df.sort_values('date')
        
        # Remove duplicates
        before_dedup = len(news_df)
        news_df = news_df.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"After deduplication: {len(news_df)} articles (removed {before_dedup - len(news_df)} duplicates)")
        
        # Filter to English articles
        if 'language' in news_df.columns and not news_df.empty:
            before_filter = len(news_df)
            
            non_english_languages = [
                'German', 'Spanish', 'French', 'Italian', 'Portuguese', 'Russian',
                'Chinese', 'Japanese', 'Korean', 'Arabic', 'Dutch', 'Polish',
                'Turkish', 'Swedish', 'Norwegian', 'Danish', 'Finnish', 'Greek',
                'Hebrew', 'Hindi', 'Indonesian', 'Malay', 'Thai', 'Vietnamese',
                'Czech', 'Hungarian', 'Romanian', 'Bulgarian', 'Croatian', 'Serbian',
                'Ukrainian', 'Nepali', 'Bengali', 'Urdu', 'Persian', 'Swahili',
                'de', 'es', 'fr', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'nl',
                'pl', 'tr', 'sv', 'no', 'da', 'fi', 'el', 'he', 'hi', 'id', 'ms',
                'th', 'vi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sr', 'uk'
            ]
            
            news_df = news_df[
                ~news_df['language'].isin(non_english_languages) |
                (news_df['language'] == '') |
                (news_df['language'].isna())
            ]
            
            filtered_count = before_filter - len(news_df)
            logger.info(f"After language filtering: {len(news_df)} articles (removed {filtered_count} non-English)")
        
        # Drop language column
        if 'language' in news_df.columns:
            news_df = news_df.drop(columns=['language'])
        
        logger.info(f"Total articles collected: {len(news_df)}")
        
        if not news_df.empty:
            logger.info(f"Date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")
        
        return news_df
        
    except Exception as e:
        logger.error(f"Error in GDELT fetching: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['date', 'title', 'url', 'source'])


def fetch_all_data(ticker):
    """
    Orchestrate fetching data from all sources with INTERNATIONAL SUPPORT.
    
    Returns:
        tuple: (price_df, news_df, social_df, company_name)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING ALL DATA FOR {ticker}")
    logger.info(f"{'='*60}\n")
    
    # Fetch price data (also gets company name)
    price_df, company_name = fetch_price_data(ticker)
    
    if price_df.empty:
        logger.error(f"Cannot proceed without price data for {ticker}")
        return price_df, pd.DataFrame(), pd.DataFrame(), company_name
    
    # Fetch news data
    news_df = fetch_news_data_gdelt(ticker, company_name)
    
    # Fetch social data
    social_df = fetch_social_data(ticker)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA FETCHING COMPLETE FOR {ticker}")
    logger.info(f"{'='*60}")
    logger.info(f"Price records: {len(price_df)}")
    logger.info(f"News articles: {len(news_df)}")
    logger.info(f"Social posts: {len(social_df)}")
    logger.info(f"{'='*60}\n")
    
    return price_df, news_df, social_df, company_name