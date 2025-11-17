"""
Sentiment Analysis Multi-Source
================================

Monitora social media e news per catturare insider information
prima che venga incorporata nelle quote dai bookmaker.

Sources:
- Twitter/X (via API o scraping)
- Reddit (r/soccer, team subreddits)
- News aggregators
- Team social media

Signals Detected:
- Injury rumors (prima dell'annuncio ufficiale)
- Lineup leaks
- Team morale/motivation
- Media pressure
- Fan sentiment
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import os

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analizza sentiment da multiple fonti per rilevare insider info.

    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_match_sentiment(team_home, team_away)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inizializza Sentiment Analyzer.

        Args:
            config: Configurazione (API keys, etc.)
        """
        self.config = config or {}

        # API clients (optional)
        self.twitter_client = None
        self.reddit_client = None

        # Hugging Face InferenceClient setup (FREE - no key required, but better with one)
        self.hf_api_key = self.config.get('huggingface_api_key') or os.getenv('HUGGINGFACE_API_KEY')
        self.hf_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.hf_client = None

        # Try to initialize InferenceClient
        try:
            from huggingface_hub import InferenceClient
            self.hf_client = InferenceClient(token=self.hf_api_key)
            if self.hf_api_key:
                logger.info("   ✓ Hugging Face client initialized with API key (higher rate limits)")
            else:
                logger.info("   ✓ Hugging Face client initialized (free tier - no key needed)")
        except ImportError:
            logger.warning("   ⚠️  huggingface_hub not installed. Sentiment analysis will use fallback mode.")
            logger.warning("      Install with: pip install huggingface_hub")

        # Initialize if credentials provided
        self._initialize_clients()

        # Keyword databases
        self.injury_keywords = [
            'infortunio', 'infortunato', 'injury', 'injured', 'hurt',
            'dolore', 'pain', 'zoppica', 'limping', 'out', 'assente',
            'indisponibile', 'unavailable', 'medical', 'treatment'
        ]

        self.lineup_keywords = [
            'formazione', 'lineup', 'starting', 'xi', 'team sheet',
            'titolare', 'starter', 'bench', 'panchina'
        ]

        self.morale_positive = [
            'motivato', 'motivated', 'confident', 'fiducioso',
            'in forma', 'on form', 'strong', 'ready', 'pronto'
        ]

        self.morale_negative = [
            'demotivato', 'pressione', 'pressure', 'crisis', 'crisi',
            'troubled', 'problems', 'problemi', 'difficoltà'
        ]

        logger.info("✅ Sentiment Analyzer initialized")

    def _analyze_text_sentiment_hf(self, text: str) -> Dict:
        """
        Analizza sentiment di un testo usando Hugging Face InferenceClient.

        Args:
            text: Testo da analizzare

        Returns:
            Dict con label (POSITIVE/NEGATIVE/NEUTRAL) e score (0-1)
        """
        # If client not available, use fallback
        if not self.hf_client:
            return self._sentiment_fallback(text)

        try:
            # Truncate text if too long (model limit ~512 tokens)
            if len(text) > 500:
                text = text[:500]

            # Call Hugging Face using InferenceClient
            result = self.hf_client.text_classification(
                text=text,
                model=self.hf_model
            )

            # Result format: [{"label": "positive", "score": 0.95}, ...]
            if isinstance(result, list) and len(result) > 0:
                # Get highest score prediction
                best = max(result, key=lambda x: x['score'])

                return {
                    'label': best['label'].upper(),
                    'score': best['score'],
                    'success': True
                }

        except Exception as e:
            error_msg = str(e).lower()

            if 'loading' in error_msg or '503' in error_msg:
                logger.warning("   ⏳ Hugging Face model loading... using fallback")
            else:
                logger.warning(f"   ⚠️  Sentiment API error: {e}")

            return self._sentiment_fallback(text)

        # Fallback
        return {'label': 'NEUTRAL', 'score': 0.5, 'success': False}

    def _sentiment_fallback(self, text: str) -> Dict:
        """
        Fallback sentiment analysis using keyword matching.

        Args:
            text: Testo da analizzare

        Returns:
            Dict con label e score stimati
        """
        text_lower = text.lower()

        # Count positive/negative keywords
        pos_count = sum(1 for kw in self.morale_positive if kw in text_lower)
        neg_count = sum(1 for kw in self.morale_negative if kw in text_lower)

        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.7, 'success': False}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': 0.7, 'success': False}
        else:
            return {'label': 'NEUTRAL', 'score': 0.6, 'success': False}

    def _initialize_clients(self):
        """Initialize social media API clients if credentials exist"""
        # Twitter/X
        twitter_token = self.config.get('twitter_bearer_token')
        if twitter_token:
            try:
                import tweepy
                self.twitter_client = tweepy.Client(bearer_token=twitter_token)
                logger.info("   ✓ Twitter client initialized")
            except ImportError:
                logger.warning("   ⚠️  tweepy not installed. Install with: pip install tweepy")

        # Reddit
        reddit_creds = self.config.get('reddit')
        if reddit_creds:
            try:
                import praw
                self.reddit_client = praw.Reddit(
                    client_id=reddit_creds.get('client_id'),
                    client_secret=reddit_creds.get('client_secret'),
                    user_agent=reddit_creds.get('user_agent', 'AsianOdds/1.0')
                )
                logger.info("   ✓ Reddit client initialized")
            except ImportError:
                logger.warning("   ⚠️  praw not installed. Install with: pip install praw")

    def analyze_match_sentiment(
        self,
        team_home: str,
        team_away: str,
        hours_before: int = 48,
        include_sources: Optional[List[str]] = None
    ) -> Dict:
        """
        Analizza sentiment per match da multiple fonti.

        Args:
            team_home: Nome squadra casa
            team_away: Nome squadra trasferta
            hours_before: Ore prima del match da analizzare
            include_sources: Lista sources da includere (default: all)

        Returns:
            Dict con analisi sentiment e signals rilevati
        """
        logger.info(f"🔍 Analyzing sentiment: {team_home} vs {team_away}")

        if include_sources is None:
            include_sources = ['mock', 'twitter', 'reddit', 'news']

        result = {
            'team_home': team_home,
            'team_away': team_away,
            'timestamp': datetime.now().isoformat(),

            # Scores (0-100)
            'insider_injury_risk_home': 0.0,
            'insider_injury_risk_away': 0.0,
            'team_morale_home': 0.0,     # -100 to +100
            'team_morale_away': 0.0,
            'media_pressure_home': 0.0,   # 0-100
            'media_pressure_away': 0.0,
            'fan_confidence_home': 50.0,  # 0-100
            'fan_confidence_away': 50.0,

            # Detected signals
            'signals': [],
            'sources_checked': include_sources,
            'total_data_points': 0
        }

        # Analyze each source
        if 'mock' in include_sources or not (self.twitter_client or self.reddit_client):
            result = self._mock_sentiment_analysis(result)

        if 'twitter' in include_sources and self.twitter_client:
            twitter_data = self._analyze_twitter(team_home, team_away, hours_before)
            result = self._merge_twitter_data(result, twitter_data)

        if 'reddit' in include_sources and self.reddit_client:
            reddit_data = self._analyze_reddit(team_home, team_away, hours_before)
            result = self._merge_reddit_data(result, reddit_data)

        if 'news' in include_sources:
            news_data = self._analyze_news(team_home, team_away, hours_before)
            result = self._merge_news_data(result, news_data)

        # Calculate overall scores
        result['overall_sentiment_home'] = self._calculate_overall_sentiment(
            result['team_morale_home'],
            result['fan_confidence_home'],
            result['media_pressure_home']
        )

        result['overall_sentiment_away'] = self._calculate_overall_sentiment(
            result['team_morale_away'],
            result['fan_confidence_away'],
            result['media_pressure_away']
        )

        # Risk assessment
        result['total_risk_score'] = (
            result['insider_injury_risk_home'] +
            result['insider_injury_risk_away']
        ) / 2

        logger.info(f"   ✓ Sentiment analyzed: {result['total_data_points']} data points")
        logger.info(f"   ✓ Signals detected: {len(result['signals'])}")

        return result

    def _analyze_twitter(
        self,
        team_home: str,
        team_away: str,
        hours_before: int
    ) -> Dict:
        """Analyze Twitter for insider info"""
        logger.info("   📱 Analyzing Twitter...")

        # Build search queries
        queries = [
            f"{team_home} injury -RT",
            f"{team_home} lineup -RT",
            f"{team_away} injury -RT",
            f"{team_away} lineup -RT"
        ]

        signals = []
        injury_mentions_home = 0
        injury_mentions_away = 0

        try:
            for query in queries:
                # Search recent tweets
                tweets = self.twitter_client.search_recent_tweets(
                    query=query,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'author_id']
                )

                if not tweets.data:
                    continue

                for tweet in tweets.data:
                    # Analyze tweet text
                    text = tweet.text.lower()

                    # Check for injury keywords
                    if any(keyword in text for keyword in self.injury_keywords):
                        # Determine which team
                        is_home = team_home.lower() in text
                        is_away = team_away.lower() in text

                        if is_home:
                            injury_mentions_home += 1
                        if is_away:
                            injury_mentions_away += 1

                        # Analyze sentiment with Hugging Face API
                        sentiment = self._analyze_text_sentiment_hf(tweet.text)

                        # High engagement = more credible
                        likes = tweet.public_metrics.get('like_count', 0)
                        retweets = tweet.public_metrics.get('retweet_count', 0)
                        engagement = likes + retweets * 2

                        if engagement > 10:  # Threshold for credibility
                            signals.append({
                                'type': 'INJURY_RUMOR',
                                'source': 'Twitter',
                                'team': team_home if is_home else team_away,
                                'text': tweet.text[:100],
                                'credibility': min(engagement / 100, 1.0),
                                'timestamp': tweet.created_at,
                                'sentiment': sentiment.get('label', 'NEUTRAL'),
                                'sentiment_score': sentiment.get('score', 0.5)
                            })

        except Exception as e:
            logger.warning(f"   ⚠️  Twitter analysis failed: {e}")

        return {
            'injury_mentions_home': injury_mentions_home,
            'injury_mentions_away': injury_mentions_away,
            'signals': signals
        }

    def _analyze_reddit(
        self,
        team_home: str,
        team_away: str,
        hours_before: int
    ) -> Dict:
        """Analyze Reddit for insider info"""
        logger.info("   🔴 Analyzing Reddit...")

        signals = []
        morale_score_home = 0
        morale_score_away = 0

        try:
            # Search in r/soccer and team-specific subreddits
            subreddits = ['soccer'] + self._get_team_subreddits(team_home, team_away)

            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)

                # Search recent posts
                search_terms = [
                    f"{team_home} {team_away}",
                    f"{team_home} lineup",
                    f"{team_away} lineup"
                ]

                for term in search_terms:
                    for post in subreddit.search(term, time_filter='day', limit=50):
                        # Analyze post title and body
                        text = (post.title + " " + post.selftext).lower()
                        full_text = post.title + " " + post.selftext

                        # Analyze sentiment with Hugging Face API
                        sentiment = self._analyze_text_sentiment_hf(full_text)
                        sentiment_label = sentiment.get('label', 'NEUTRAL')
                        sentiment_score = sentiment.get('score', 0.5)

                        # Check keywords
                        if any(kw in text for kw in self.injury_keywords):
                            signals.append({
                                'type': 'INJURY_RUMOR',
                                'source': 'Reddit',
                                'text': post.title,
                                'upvotes': post.score,
                                'credibility': min(post.score / 100, 1.0),
                                'sentiment': sentiment_label,
                                'sentiment_score': sentiment_score
                            })

                        # Morale sentiment from API
                        if team_home.lower() in text:
                            if sentiment_label == 'POSITIVE':
                                morale_score_home += sentiment_score * 10
                            elif sentiment_label == 'NEGATIVE':
                                morale_score_home -= sentiment_score * 10

                        if team_away.lower() in text:
                            if sentiment_label == 'POSITIVE':
                                morale_score_away += sentiment_score * 10
                            elif sentiment_label == 'NEGATIVE':
                                morale_score_away -= sentiment_score * 10

        except Exception as e:
            logger.warning(f"   ⚠️  Reddit analysis failed: {e}")

        return {
            'morale_score_home': morale_score_home,
            'morale_score_away': morale_score_away,
            'signals': signals
        }

    def _analyze_news(
        self,
        team_home: str,
        team_away: str,
        hours_before: int
    ) -> Dict:
        """Analyze news articles (via Google News RSS - FREE)"""
        logger.info("   📰 Analyzing news...")

        signals = []
        news_sentiment_home = 0
        news_sentiment_away = 0

        try:
            # Google News RSS (free, no API key needed)
            import feedparser

            # Search queries for both teams
            queries = [
                f"{team_home} football",
                f"{team_away} football"
            ]

            for query in queries:
                # Google News RSS URL
                rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en&gl=US&ceid=US:en"

                # Parse RSS feed
                feed = feedparser.parse(rss_url)

                # Analyze recent articles (last 48h)
                for entry in feed.entries[:10]:  # Limit to 10 articles per team
                    title = entry.get('title', '')
                    description = entry.get('summary', '')
                    full_text = title + " " + description

                    # Analyze sentiment with Hugging Face
                    sentiment = self._analyze_text_sentiment_hf(full_text)
                    sentiment_label = sentiment.get('label', 'NEUTRAL')
                    sentiment_score = sentiment.get('score', 0.5)

                    # Check if about team home or away
                    text_lower = full_text.lower()
                    is_home = team_home.lower() in text_lower
                    is_away = team_away.lower() in text_lower

                    # Accumulate sentiment scores
                    if is_home:
                        if sentiment_label == 'POSITIVE':
                            news_sentiment_home += sentiment_score
                        elif sentiment_label == 'NEGATIVE':
                            news_sentiment_home -= sentiment_score

                    if is_away:
                        if sentiment_label == 'POSITIVE':
                            news_sentiment_away += sentiment_score
                        elif sentiment_label == 'NEGATIVE':
                            news_sentiment_away -= sentiment_score

                    # Check for injury news
                    if any(kw in text_lower for kw in self.injury_keywords):
                        signals.append({
                            'type': 'INJURY_NEWS',
                            'source': 'Google News',
                            'team': team_home if is_home else (team_away if is_away else 'Unknown'),
                            'text': title[:100],
                            'link': entry.get('link', ''),
                            'sentiment': sentiment_label,
                            'sentiment_score': sentiment_score,
                            'credibility': 0.8  # News generally more credible than social
                        })

        except ImportError:
            logger.warning("   ⚠️  feedparser not installed. Install with: pip install feedparser")
        except Exception as e:
            logger.warning(f"   ⚠️  News analysis failed: {e}")

        return {
            'news_sentiment_home': news_sentiment_home,
            'news_sentiment_away': news_sentiment_away,
            'signals': signals
        }

    def _mock_sentiment_analysis(self, result: Dict) -> Dict:
        """Mock sentiment analysis for testing"""
        logger.info("   🤖 Using mock sentiment data")

        # Simulate some signals
        import random

        # Random injury risk (usually low)
        result['insider_injury_risk_home'] = random.uniform(0, 30)
        result['insider_injury_risk_away'] = random.uniform(0, 30)

        # Random morale (usually positive)
        result['team_morale_home'] = random.uniform(-20, 50)
        result['team_morale_away'] = random.uniform(-20, 50)

        # Media pressure (higher for big teams)
        result['media_pressure_home'] = random.uniform(30, 70)
        result['media_pressure_away'] = random.uniform(30, 70)

        # Fan confidence
        result['fan_confidence_home'] = random.uniform(40, 80)
        result['fan_confidence_away'] = random.uniform(40, 80)

        # Simulate occasional signal
        if random.random() < 0.3:  # 30% chance
            result['signals'].append({
                'type': 'INJURY_RUMOR',
                'source': 'Mock',
                'team': result['team_home'],
                'text': 'Possible injury concern mentioned in social media',
                'credibility': 0.6
            })

        result['total_data_points'] = 50  # Mock count

        return result

    def _merge_twitter_data(self, result: Dict, twitter_data: Dict) -> Dict:
        """Merge Twitter analysis into result"""
        # Injury risk increases with mentions
        result['insider_injury_risk_home'] += min(twitter_data['injury_mentions_home'] * 5, 50)
        result['insider_injury_risk_away'] += min(twitter_data['injury_mentions_away'] * 5, 50)

        # Add signals
        result['signals'].extend(twitter_data['signals'])
        result['total_data_points'] += len(twitter_data['signals'])

        return result

    def _merge_reddit_data(self, result: Dict, reddit_data: Dict) -> Dict:
        """Merge Reddit analysis into result"""
        # Morale score
        result['team_morale_home'] += reddit_data['morale_score_home'] * 2
        result['team_morale_away'] += reddit_data['morale_score_away'] * 2

        # Add signals
        result['signals'].extend(reddit_data['signals'])
        result['total_data_points'] += len(reddit_data['signals'])

        return result

    def _merge_news_data(self, result: Dict, news_data: Dict) -> Dict:
        """Merge news analysis into result"""
        # Add news sentiment to morale
        result['team_morale_home'] += news_data.get('news_sentiment_home', 0) * 5
        result['team_morale_away'] += news_data.get('news_sentiment_away', 0) * 5

        # Add signals
        result['signals'].extend(news_data['signals'])
        result['total_data_points'] += len(news_data['signals'])

        return result

    def _calculate_overall_sentiment(
        self,
        morale: float,
        fan_confidence: float,
        media_pressure: float
    ) -> float:
        """
        Calculate overall sentiment score (-100 to +100).

        Weighted combination of factors.
        """
        # Normalize and combine
        score = (
            morale * 0.4 +
            (fan_confidence - 50) * 0.4 +  # Center at 50
            (50 - media_pressure) * 0.2  # High pressure = negative
        )

        # Clamp
        return max(-100, min(100, score))

    def _get_team_subreddits(self, team_home: str, team_away: str) -> List[str]:
        """Get team-specific subreddit names"""
        # Mapping of teams to subreddits (would be more comprehensive)
        subreddit_map = {
            'inter': 'FCInterMilan',
            'milan': 'ACMilan',
            'juventus': 'Juve',
            'napoli': 'sscnapoli',
            'roma': 'ASRoma',
            'lazio': 'lazio'
        }

        subs = []
        for team in [team_home, team_away]:
            team_lower = team.lower()
            for key, sub in subreddit_map.items():
                if key in team_lower:
                    subs.append(sub)

        return subs


def adjust_prediction_with_sentiment(
    base_probability: float,
    sentiment_result: Dict,
    team: str = 'home'
) -> Tuple[float, List[str]]:
    """
    Adjust prediction probability based on sentiment analysis.

    Args:
        base_probability: Original probability
        sentiment_result: Result from analyze_match_sentiment()
        team: 'home' or 'away'

    Returns:
        Tuple of (adjusted_probability, reasons)
    """
    adjusted_prob = base_probability
    reasons = []

    # Injury risk adjustment
    injury_key = f'insider_injury_risk_{team}'
    injury_risk = sentiment_result.get(injury_key, 0)

    if injury_risk > 60:
        # High injury risk -> reduce probability
        adjustment = -0.10  # -10%
        adjusted_prob += adjustment
        reasons.append(f"⚠️ High injury risk detected ({injury_risk:.0f}%): {adjustment:+.1%}")

    elif injury_risk > 40:
        adjustment = -0.05  # -5%
        adjusted_prob += adjustment
        reasons.append(f"⚠️ Moderate injury risk ({injury_risk:.0f}%): {adjustment:+.1%}")

    # Morale adjustment
    morale_key = f'team_morale_{team}'
    morale = sentiment_result.get(morale_key, 0)

    if morale > 30:
        adjustment = +0.03  # +3%
        adjusted_prob += adjustment
        reasons.append(f"✅ Positive morale: {adjustment:+.1%}")
    elif morale < -30:
        adjustment = -0.03  # -3%
        adjusted_prob += adjustment
        reasons.append(f"⚠️ Negative morale: {adjustment:+.1%}")

    # Overall sentiment
    sentiment_key = f'overall_sentiment_{team}'
    sentiment = sentiment_result.get(sentiment_key, 0)

    if abs(sentiment) > 40:
        adjustment = sentiment / 1000  # Max ±4%
        adjusted_prob += adjustment
        reasons.append(f"Overall sentiment: {adjustment:+.1%}")

    # Clamp to valid range
    adjusted_prob = max(0.01, min(0.99, adjusted_prob))

    return adjusted_prob, reasons


if __name__ == "__main__":
    # Test Sentiment Analyzer
    logging.basicConfig(level=logging.INFO)

    print("Testing Sentiment Analyzer (Mock Mode)...")
    print("=" * 70)

    analyzer = SentimentAnalyzer()

    # Analyze match sentiment
    result = analyzer.analyze_match_sentiment(
        team_home="Inter",
        team_away="Napoli",
        hours_before=48
    )

    print("\n📊 SENTIMENT ANALYSIS RESULT:")
    print(f"Team Home: {result['team_home']}")
    print(f"Team Away: {result['team_away']}")
    print(f"\n🏠 HOME TEAM:")
    print(f"   Injury Risk: {result['insider_injury_risk_home']:.1f}%")
    print(f"   Morale: {result['team_morale_home']:+.1f}")
    print(f"   Fan Confidence: {result['fan_confidence_home']:.1f}%")
    print(f"   Overall Sentiment: {result['overall_sentiment_home']:+.1f}")

    print(f"\n✈️  AWAY TEAM:")
    print(f"   Injury Risk: {result['insider_injury_risk_away']:.1f}%")
    print(f"   Morale: {result['team_morale_away']:+.1f}")
    print(f"   Fan Confidence: {result['fan_confidence_away']:.1f}%")
    print(f"   Overall Sentiment: {result['overall_sentiment_away']:+.1f}")

    print(f"\n🚨 SIGNALS DETECTED: {len(result['signals'])}")
    for signal in result['signals']:
        print(f"   - {signal['type']}: {signal['text'][:60]}... (source: {signal['source']})")

    # Test adjustment
    print("\n" + "=" * 70)
    print("Testing probability adjustment...")

    base_prob = 0.60
    adjusted_prob, reasons = adjust_prediction_with_sentiment(base_prob, result, 'home')

    print(f"\nBase Probability: {base_prob:.1%}")
    print(f"Adjusted Probability: {adjusted_prob:.1%}")
    print(f"Adjustment: {adjusted_prob - base_prob:+.1%}")
    print(f"\nReasons:")
    for reason in reasons:
        print(f"   {reason}")

    print("\n" + "=" * 70)
    print("✅ Sentiment Analyzer test completed!")
