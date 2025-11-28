"""
Sistema Analisi News/Social Media
==================================

Scraping news sportive e analisi sentiment da Twitter/Reddit.
Alert quando ci sono notizie importanti (infortuni, cambi allenatore).
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Notizia sportiva"""
    title: str
    content: str
    source: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment: Optional[float] = None  # -1 to 1
    importance: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    teams_mentioned: List[str] = None
    keywords: List[str] = None


class NewsSentimentAnalyzer:
    """
    Analizza news e sentiment da varie fonti.
    """
    
    def __init__(self, newsapi_key: Optional[str] = None, sentiment_analyzer=None):
        """
        Args:
            newsapi_key: API key per NewsAPI.org (opzionale, 100 richieste/giorno gratis)
            sentiment_analyzer: Istanza di SentimentAnalyzer (già implementato)
        """
        self.newsapi_key = newsapi_key
        self.sentiment_analyzer = sentiment_analyzer
        self.processed_news: set = set()
    
    def fetch_sports_news(
        self,
        query: str = "football",
        language: str = "en",
        max_results: int = 10
    ) -> List[NewsItem]:
        """
        Recupera news sportive da NewsAPI.org.
        
        Args:
            query: Query di ricerca
            language: Lingua (en, it, etc.)
            max_results: Numero massimo risultati
        """
        news_items = []
        
        if not self.newsapi_key:
            logger.debug("⚠️  NewsAPI key non configurata, skip news fetching")
            return news_items
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': language,
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            for article in data.get('articles', []):
                # Evita duplicati
                article_id = article.get('url', article.get('title', ''))
                if article_id in self.processed_news:
                    continue
                
                # Parse data pubblicazione
                published_at = None
                if article.get('publishedAt'):
                    try:
                        published_at = datetime.fromisoformat(
                            article['publishedAt'].replace('Z', '+00:00')
                        )
                    except:
                        pass
                
                news_item = NewsItem(
                    title=article.get('title', ''),
                    content=article.get('description', '') or article.get('content', ''),
                    source=article.get('source', {}).get('name', 'Unknown'),
                    url=article.get('url'),
                    published_at=published_at
                )
                
                # Analizza sentiment se disponibile
                if self.sentiment_analyzer and news_item.content:
                    try:
                        sentiment_result = self.sentiment_analyzer.analyze(news_item.content)
                        if sentiment_result:
                            news_item.sentiment = sentiment_result.get('overall_sentiment', 0)
                    except:
                        pass
                
                # Estrai team menzionati e keywords
                news_item.teams_mentioned = self._extract_teams(news_item.content)
                news_item.keywords = self._extract_keywords(news_item.content)
                
                # Determina importanza
                news_item.importance = self._determine_importance(news_item)
                
                news_items.append(news_item)
                self.processed_news.add(article_id)
            
            logger.info(f"✅ Recuperate {len(news_items)} news")
            
        except Exception as e:
            logger.warning(f"⚠️  Errore fetch news: {e}")
        
        return news_items
    
    def _extract_teams(self, text: str) -> List[str]:
        """Estrae nomi team dal testo (semplificato)"""
        # Lista team comuni (da espandere)
        common_teams = [
            'Inter', 'Milan', 'Juventus', 'Napoli', 'Roma', 'Lazio',
            'Real Madrid', 'Barcelona', 'Atletico Madrid',
            'Manchester United', 'Manchester City', 'Liverpool', 'Chelsea', 'Arsenal',
            'Bayern Munich', 'Borussia Dortmund',
            'PSG', 'Lyon', 'Marseille'
        ]
        
        mentioned = []
        text_lower = text.lower()
        
        for team in common_teams:
            if team.lower() in text_lower:
                mentioned.append(team)
        
        return mentioned
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Estrae keywords importanti"""
        keywords = []
        text_lower = text.lower()
        
        important_keywords = [
            'injury', 'infortunio', 'injured', 'infortunato',
            'transfer', 'trasferimento', 'signed', 'firmato',
            'coach', 'allenatore', 'manager', 'sacked', 'licenziato',
            'suspension', 'squalifica', 'banned', 'squalificato',
            'goal', 'gol', 'score', 'risultato'
        ]
        
        for keyword in important_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _determine_importance(self, news: NewsItem) -> str:
        """Determina importanza notizia"""
        importance_score = 0
        
        # Keywords importanti
        critical_keywords = ['injury', 'infortunio', 'sacked', 'licenziato', 'transfer', 'trasferimento']
        for keyword in critical_keywords:
            if keyword in news.content.lower() or keyword in news.title.lower():
                importance_score += 2
        
        # Sentiment estremo
        if news.sentiment:
            if abs(news.sentiment) > 0.7:
                importance_score += 1
        
        # Team menzionati
        if news.teams_mentioned:
            importance_score += 1
        
        if importance_score >= 3:
            return 'CRITICAL'
        elif importance_score >= 2:
            return 'HIGH'
        elif importance_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_relevant_news(
        self,
        team_name: str,
        hours: int = 24
    ) -> List[NewsItem]:
        """Ottiene news rilevanti per un team"""
        all_news = self.fetch_sports_news(query=team_name)
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        relevant = [
            news for news in all_news
            if (not news.published_at or news.published_at > cutoff) and
               (team_name.lower() in news.title.lower() or
                team_name.lower() in news.content.lower() or
                team_name in (news.teams_mentioned or []))
        ]
        
        # Ordina per importanza
        importance_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        relevant.sort(key=lambda x: importance_order.get(x.importance, 3))
        
        return relevant
    
    def format_news_alert(self, news: NewsItem) -> str:
        """Formatta messaggio alert per notizia"""
        emoji = {
            'CRITICAL': '🚨',
            'HIGH': '🔥',
            'MEDIUM': '⚡',
            'LOW': '📰'
        }.get(news.importance, '📰')
        
        message = f"{emoji} {news.importance} - News Importante\n\n"
        message += f"📰 {news.title}\n\n"
        message += f"{news.content[:200]}...\n\n"
        message += f"📊 Fonte: {news.source}\n"
        
        if news.teams_mentioned:
            message += f"👥 Team: {', '.join(news.teams_mentioned)}\n"
        
        if news.sentiment is not None:
            sentiment_emoji = '🟢' if news.sentiment > 0.3 else '🔴' if news.sentiment < -0.3 else '🟡'
            message += f"{sentiment_emoji} Sentiment: {news.sentiment:.2f}\n"
        
        if news.url:
            message += f"🔗 {news.url}\n"
        
        return message

