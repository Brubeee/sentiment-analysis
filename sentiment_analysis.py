"""
Sentiment Analysis module - FIXED VERSION
Handles AI-powered sentiment analysis with proper NaN handling.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import config
import numpy as np
from datetime import datetime
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

# Global model cache
_news_model = None
_social_model = None
_models_loaded = False


def clean_text(text):
    """Clean and preprocess text for sentiment analysis."""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove URLs
    import re
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"$%]', ' ', text)
    
    # Limit length
    if len(text) > 2000:
        text = text[:2000]
    
    return text.strip()


def load_sentiment_models():
    """Load sentiment models with caching."""
    global _news_model, _social_model, _models_loaded
    
    if _models_loaded:
        return _news_model, _social_model
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    
    # Load news model (FinBERT)
    if _news_model is None:
        logger.info(f"Loading news sentiment model: {config.NEWS_SENTIMENT_MODEL} on {device_name}")
        try:
            _news_model = pipeline(
                "sentiment-analysis",
                model=config.NEWS_SENTIMENT_MODEL,
                tokenizer=config.NEWS_SENTIMENT_MODEL,
                device=device,
                max_length=512,
                truncation=True,
                batch_size=config.SENTIMENT_BATCH_SIZE
            )
            logger.info("✓ News sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading news model: {str(e)}")
            _news_model = None
    
    # Load social model (RoBERTa)
    if _social_model is None:
        logger.info(f"Loading social sentiment model: {config.SOCIAL_SENTIMENT_MODEL} on {device_name}")
        try:
            _social_model = pipeline(
                "sentiment-analysis",
                model=config.SOCIAL_SENTIMENT_MODEL,
                tokenizer=config.SOCIAL_SENTIMENT_MODEL,
                device=device,
                max_length=512,
                truncation=True,
                batch_size=config.SENTIMENT_BATCH_SIZE
            )
            logger.info("✓ Social sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading social model: {str(e)}")
            _social_model = None
    
    _models_loaded = True
    return _news_model, _social_model


def unload_models():
    """Unload models from memory."""
    global _news_model, _social_model, _models_loaded
    
    _news_model = None
    _social_model = None
    _models_loaded = False
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Models unloaded from memory")


def normalize_sentiment_score(result, model_type='finbert'):
    """
    Normalize sentiment scores to -1 to +1 scale.
    Returns 0.0 for any errors or invalid data.
    """
    try:
        label = result['label'].lower()
        score = result['score']
        
        # Validate score is a number
        if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
            return 0.0
        
        if model_type == 'finbert':
            if 'positive' in label:
                return float(score)
            elif 'negative' in label:
                return float(-score)
            else:  # neutral
                return 0.0
        
        elif model_type == 'roberta':
            if 'label_2' in label or 'positive' in label:
                return float(score)
            elif 'label_0' in label or 'negative' in label:
                return float(-score)
            else:  # label_1 or neutral
                return 0.0
        
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error normalizing score: {str(e)}")
        return 0.0


def analyze_texts_batch(texts, model, model_type='finbert', batch_size=None, show_progress=True):
    """Analyze sentiment for a batch of texts with progress tracking."""
    if not texts or model is None:
        logger.warning(f"No texts to analyze or model is None")
        return [0.0] * len(texts)
    
    batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
    scores = []
    
    # Clean texts first
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(cleaned_texts) if text]
    valid_texts = [cleaned_texts[i] for i in valid_indices]
    
    if not valid_texts:
        logger.warning("No valid texts after cleaning")
        return [0.0] * len(texts)
    
    # Process in batches with progress bar
    num_batches = (len(valid_texts) + batch_size - 1) // batch_size
    
    iterator = range(0, len(valid_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=f"Analyzing {model_type}", unit="batch")
    
    valid_scores = []
    
    for i in iterator:
        batch = valid_texts[i:i + batch_size]
        
        try:
            results = model(batch)
            batch_scores = [
                normalize_sentiment_score(result, model_type)
                for result in results
            ]
            valid_scores.extend(batch_scores)
            
        except Exception as e:
            logger.warning(f"Error processing batch at index {i}: {str(e)}")
            valid_scores.extend([0.0] * len(batch))
    
    # Map scores back to original indices
    all_scores = [0.0] * len(texts)
    for idx, score in zip(valid_indices, valid_scores):
        all_scores[idx] = score
    
    return all_scores


def aggregate_daily_sentiment(df, date_col='date', score_col='sentiment_score', weight_col=None):
    """Aggregate individual sentiment scores into daily averages."""
    if df.empty:
        return pd.DataFrame(columns=['date', 'daily_sentiment', 'count'])
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['date_only'] = df[date_col].dt.date
    
    # Calculate weighted or simple average
    if weight_col and weight_col in df.columns:
        df['weighted_score'] = df[score_col] * df[weight_col].clip(lower=1)
        
        daily = df.groupby('date_only').agg({
            'weighted_score': 'sum',
            weight_col: 'sum',
            score_col: ['std', 'count']
        }).reset_index()
        
        daily.columns = ['date', 'weighted_sum', 'weight_sum', 'sentiment_std', 'count']
        daily['daily_sentiment'] = daily['weighted_sum'] / daily['weight_sum']
        daily = daily.drop(columns=['weighted_sum', 'weight_sum'])
    else:
        daily = df.groupby('date_only').agg({
            score_col: ['mean', 'std', 'count']
        }).reset_index()
        
        daily.columns = ['date', 'daily_sentiment', 'sentiment_std', 'count']
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    
    # CRITICAL FIX: Replace any NaN values with 0
    daily['daily_sentiment'] = daily['daily_sentiment'].fillna(0.0)
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0.0)
    
    return daily


def analyze_news_sentiment(news_df):
    """Analyze sentiment of news articles using FinBERT."""
    if news_df.empty:
        logger.warning("No news data to analyze")
        return pd.DataFrame(columns=['date', 'daily_sentiment', 'count'])
    
    logger.info(f"Analyzing sentiment for {len(news_df)} news articles...")
    
    news_model, _ = load_sentiment_models()
    
    if news_model is None:
        logger.error("Failed to load news sentiment model")
        return pd.DataFrame(columns=['date', 'daily_sentiment', 'count'])
    
    texts = news_df['title'].fillna('').astype(str).tolist()
    
    sentiment_scores = analyze_texts_batch(
        texts, 
        news_model, 
        model_type='finbert',
        show_progress=True
    )
    
    news_df_copy = news_df.copy()
    news_df_copy['sentiment_score'] = sentiment_scores
    
    daily_news_sentiment = aggregate_daily_sentiment(news_df_copy)
    
    avg_sentiment = daily_news_sentiment['daily_sentiment'].mean()
    logger.info(f"Generated daily news sentiment for {len(daily_news_sentiment)} days")
    logger.info(f"Average news sentiment: {avg_sentiment:.4f}")
    
    return daily_news_sentiment


def analyze_social_sentiment(social_df):
    """Analyze sentiment of social media posts using RoBERTa."""
    if social_df.empty:
        logger.warning("No social data to analyze")
        return pd.DataFrame(columns=['date', 'daily_sentiment', 'count'])
    
    logger.info(f"Analyzing sentiment for {len(social_df)} social posts...")
    
    _, social_model = load_sentiment_models()
    
    if social_model is None:
        logger.error("Failed to load social sentiment model")
        return pd.DataFrame(columns=['date', 'daily_sentiment', 'count'])
    
    texts = social_df['text'].fillna('').astype(str).tolist()
    
    sentiment_scores = analyze_texts_batch(
        texts, 
        social_model, 
        model_type='roberta',
        show_progress=True
    )
    
    social_df_copy = social_df.copy()
    social_df_copy['sentiment_score'] = sentiment_scores
    
    weight_col = 'score' if 'score' in social_df_copy.columns else None
    daily_social_sentiment = aggregate_daily_sentiment(
        social_df_copy, 
        weight_col=weight_col
    )
    
    avg_sentiment = daily_social_sentiment['daily_sentiment'].mean()
    logger.info(f"Generated daily social sentiment for {len(daily_social_sentiment)} days")
    logger.info(f"Average social sentiment: {avg_sentiment:.4f}")
    
    return daily_social_sentiment


def combine_sentiment_scores(news_sentiment_df, social_sentiment_df):
    """
    FIXED: Combine news and social sentiment with proper NaN handling.
    """
    logger.info("Combining news and social sentiment scores...")
    
    if news_sentiment_df.empty and social_sentiment_df.empty:
        logger.warning("No sentiment data to combine")
        return pd.DataFrame(columns=['date', 'combined_sentiment'])
    
    # Prepare news data
    if not news_sentiment_df.empty:
        news_sentiment_df = news_sentiment_df.copy()
        news_sentiment_df = news_sentiment_df.rename(columns={'daily_sentiment': 'news_sentiment'})
        news_sentiment_df = news_sentiment_df[['date', 'news_sentiment']]
    
    # Prepare social data
    if not social_sentiment_df.empty:
        social_sentiment_df = social_sentiment_df.copy()
        social_sentiment_df = social_sentiment_df.rename(columns={'daily_sentiment': 'social_sentiment'})
        social_sentiment_df = social_sentiment_df[['date', 'social_sentiment']]
    
    # Merge on date (outer join to keep all dates)
    if not news_sentiment_df.empty and not social_sentiment_df.empty:
        combined_df = pd.merge(
            news_sentiment_df,
            social_sentiment_df,
            on='date',
            how='outer'
        )
    elif not news_sentiment_df.empty:
        combined_df = news_sentiment_df.copy()
        combined_df['social_sentiment'] = 0.0
    else:
        combined_df = social_sentiment_df.copy()
        combined_df['news_sentiment'] = 0.0
    
    # Sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # CRITICAL FIX: Use .ffill() instead of deprecated fillna(method='ffill')
    # Then fill remaining NaN with 0
    combined_df['news_sentiment'] = combined_df['news_sentiment'].ffill().fillna(0.0)
    combined_df['social_sentiment'] = combined_df['social_sentiment'].ffill().fillna(0.0)
    
    # Calculate weighted combined sentiment
    combined_df['combined_sentiment'] = (
        config.NEWS_WEIGHT * combined_df['news_sentiment'] +
        config.SOCIAL_WEIGHT * combined_df['social_sentiment']
    )
    
    # CRITICAL FIX: Final safety check - replace any remaining NaN
    combined_df['combined_sentiment'] = combined_df['combined_sentiment'].fillna(0.0)
    
    # Validate no NaN values remain
    if combined_df['combined_sentiment'].isna().any():
        logger.error("WARNING: NaN values still present after combining!")
        combined_df['combined_sentiment'] = combined_df['combined_sentiment'].fillna(0.0)
    
    logger.info(f"Generated combined sentiment for {len(combined_df)} days")
    
    if len(combined_df) > 0:
        logger.info(f"Average combined sentiment: {combined_df['combined_sentiment'].mean():.4f}")
        logger.info(f"Sentiment range: [{combined_df['combined_sentiment'].min():.4f}, {combined_df['combined_sentiment'].max():.4f}]")
    
    return combined_df


def analyze_sentiment(news_df, social_df):
    """Main sentiment analysis function."""
    logger.info("\n" + "="*60)
    logger.info("SENTIMENT ANALYSIS PIPELINE")
    logger.info("="*60 + "\n")
    
    start_time = datetime.now()
    
    try:
        news_sentiment_df = analyze_news_sentiment(news_df)
        social_sentiment_df = analyze_social_sentiment(social_df)
        combined_sentiment_df = combine_sentiment_scores(news_sentiment_df, social_sentiment_df)
        
        if not combined_sentiment_df.empty:
            logger.info("\n" + "-"*60)
            logger.info("SENTIMENT ANALYSIS SUMMARY")
            logger.info("-"*60)
            logger.info(f"Total days covered: {len(combined_sentiment_df)}")
            logger.info(f"Date range: {combined_sentiment_df['date'].min().date()} to {combined_sentiment_df['date'].max().date()}")
            logger.info(f"Average combined sentiment: {combined_sentiment_df['combined_sentiment'].mean():.4f}")
            logger.info(f"Sentiment std deviation: {combined_sentiment_df['combined_sentiment'].std():.4f}")
            
            positive_days = (combined_sentiment_df['combined_sentiment'] > 0.1).sum()
            negative_days = (combined_sentiment_df['combined_sentiment'] < -0.1).sum()
            neutral_days = len(combined_sentiment_df) - positive_days - negative_days
            
            logger.info(f"\nSentiment distribution:")
            logger.info(f"  Positive days: {positive_days} ({100*positive_days/len(combined_sentiment_df):.1f}%)")
            logger.info(f"  Neutral days:  {neutral_days} ({100*neutral_days/len(combined_sentiment_df):.1f}%)")
            logger.info(f"  Negative days: {negative_days} ({100*negative_days/len(combined_sentiment_df):.1f}%)")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nSentiment analysis completed in {elapsed:.1f} seconds")
        logger.info("="*60 + "\n")
        
        return combined_sentiment_df
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['date', 'combined_sentiment'])