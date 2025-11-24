"""
Custom Exceptions
=================

Domain-specific exceptions for the AsianOdds betting system.
Use these instead of generic Exception to improve error handling and debugging.
"""


# ============================================================
# BASE EXCEPTIONS
# ============================================================

class AsianOddsError(Exception):
    """Base exception for all AsianOdds errors"""
    pass


# ============================================================
# API EXCEPTIONS
# ============================================================

class APIError(AsianOddsError):
    """Base exception for API-related errors"""
    pass


class APIConnectionError(APIError):
    """Raised when API connection fails"""
    pass


class APITimeoutError(APIError):
    """Raised when API request times out"""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds until rate limit resets


class APIAuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass


class APIQuotaExceededError(APIError):
    """Raised when API daily quota is exceeded"""
    pass


class APIInvalidResponseError(APIError):
    """Raised when API returns invalid or unexpected response"""
    pass


class APIKeyMissingError(APIError):
    """Raised when required API key is not configured"""
    pass


# ============================================================
# DATA VALIDATION EXCEPTIONS
# ============================================================

class DataValidationError(AsianOddsError):
    """Base exception for data validation errors"""
    pass


class InvalidMatchDataError(DataValidationError):
    """Raised when match data is invalid or incomplete"""
    pass


class InvalidOddsError(DataValidationError):
    """Raised when odds data is invalid"""
    pass


class InvalidProbabilityError(DataValidationError):
    """Raised when probability value is invalid (not in [0, 1])"""
    pass


class InsufficientDataError(DataValidationError):
    """Raised when insufficient data is available for analysis"""
    pass


class DataQualityError(DataValidationError):
    """Raised when data quality is below acceptable threshold"""
    pass


# ============================================================
# BETTING EXCEPTIONS
# ============================================================

class BettingError(AsianOddsError):
    """Base exception for betting-related errors"""
    pass


class InvalidStakeError(BettingError):
    """Raised when stake amount is invalid"""
    pass


class BankrollExceededError(BettingError):
    """Raised when bet would exceed bankroll limits"""
    pass


class RiskLimitExceededError(BettingError):
    """Raised when bet would exceed risk management limits"""
    pass


class InvalidMarketError(BettingError):
    """Raised when betting market is invalid or unsupported"""
    pass


class BetPlacementError(BettingError):
    """Raised when bet placement fails"""
    pass


class NoValueBetError(BettingError):
    """Raised when no value bet opportunities are found"""
    pass


# ============================================================
# AI/ML EXCEPTIONS
# ============================================================

class AIError(AsianOddsError):
    """Base exception for AI/ML errors"""
    pass


class ModelNotFoundError(AIError):
    """Raised when ML model file is not found"""
    pass


class ModelLoadError(AIError):
    """Raised when ML model fails to load"""
    pass


class PredictionError(AIError):
    """Raised when model prediction fails"""
    pass


class CalibrationError(AIError):
    """Raised when probability calibration fails"""
    pass


class ModelNotTrainedError(AIError):
    """Raised when attempting to use untrained model"""
    pass


class InsufficientTrainingDataError(AIError):
    """Raised when insufficient data for model training"""
    pass


# ============================================================
# CONFIGURATION EXCEPTIONS
# ============================================================

class ConfigurationError(AsianOddsError):
    """Base exception for configuration errors"""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid"""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing"""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails"""
    pass


# ============================================================
# DATABASE EXCEPTIONS (imported from database_manager)
# ============================================================

class DatabaseError(AsianOddsError):
    """Base exception for database errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails"""
    pass


# ============================================================
# NOTIFICATION EXCEPTIONS
# ============================================================

class NotificationError(AsianOddsError):
    """Base exception for notification errors"""
    pass


class TelegramError(NotificationError):
    """Raised when Telegram notification fails"""
    pass


class TelegramRateLimitError(TelegramError):
    """Raised when Telegram rate limit is exceeded"""
    pass


# ============================================================
# LIVE BETTING EXCEPTIONS
# ============================================================

class LiveBettingError(AsianOddsError):
    """Base exception for live betting errors"""
    pass


class MatchNotLiveError(LiveBettingError):
    """Raised when attempting live betting on non-live match"""
    pass


class LiveDataUnavailableError(LiveBettingError):
    """Raised when live match data is unavailable"""
    pass


class LiveDataStaleError(LiveBettingError):
    """Raised when live data is too old"""
    pass


# ============================================================
# ODDS TRACKING EXCEPTIONS
# ============================================================

class OddsTrackingError(AsianOddsError):
    """Base exception for odds tracking errors"""
    pass


class OddsMovementError(OddsTrackingError):
    """Raised when odds movement detection fails"""
    pass


class OddsHistoryError(OddsTrackingError):
    """Raised when odds history is insufficient"""
    pass


# ============================================================
# SYSTEM EXCEPTIONS
# ============================================================

class SystemError(AsianOddsError):
    """Base exception for system-level errors"""
    pass


class InitializationError(SystemError):
    """Raised when system initialization fails"""
    pass


class ResourceError(SystemError):
    """Raised when system resource is unavailable"""
    pass


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def handle_api_error(e: Exception, provider: str = "Unknown") -> APIError:
    """
    Convert generic exception to appropriate APIError subclass.

    Args:
        e: Original exception
        provider: API provider name

    Returns:
        Appropriate APIError subclass
    """
    error_msg = str(e).lower()

    if "timeout" in error_msg:
        return APITimeoutError(f"{provider} API timeout: {e}")
    elif "rate limit" in error_msg or "429" in error_msg:
        return APIRateLimitError(f"{provider} rate limit exceeded: {e}")
    elif "auth" in error_msg or "401" in error_msg or "403" in error_msg:
        return APIAuthenticationError(f"{provider} authentication failed: {e}")
    elif "quota" in error_msg:
        return APIQuotaExceededError(f"{provider} quota exceeded: {e}")
    elif "connection" in error_msg:
        return APIConnectionError(f"{provider} connection failed: {e}")
    else:
        return APIError(f"{provider} API error: {e}")


def validate_probability(prob: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability [0, 1].

    Args:
        prob: Probability value to validate
        name: Name of the probability for error message

    Raises:
        InvalidProbabilityError: If probability is invalid
    """
    if not isinstance(prob, (int, float)):
        raise InvalidProbabilityError(f"{name} must be numeric, got {type(prob)}")

    if not (0 <= prob <= 1):
        raise InvalidProbabilityError(f"{name} must be in [0, 1], got {prob}")


def validate_odds(odds: float, name: str = "odds") -> None:
    """
    Validate that odds value is valid (> 1.0).

    Args:
        odds: Odds value to validate
        name: Name of the odds for error message

    Raises:
        InvalidOddsError: If odds are invalid
    """
    if not isinstance(odds, (int, float)):
        raise InvalidOddsError(f"{name} must be numeric, got {type(odds)}")

    if odds <= 1.0:
        raise InvalidOddsError(f"{name} must be > 1.0, got {odds}")


def validate_stake(stake: float, min_stake: float, max_stake: float) -> None:
    """
    Validate that stake is within allowed range.

    Args:
        stake: Stake amount to validate
        min_stake: Minimum allowed stake
        max_stake: Maximum allowed stake

    Raises:
        InvalidStakeError: If stake is invalid
    """
    if not isinstance(stake, (int, float)):
        raise InvalidStakeError(f"Stake must be numeric, got {type(stake)}")

    if stake < 0:
        raise InvalidStakeError(f"Stake cannot be negative, got {stake}")

    if stake < min_stake:
        raise InvalidStakeError(f"Stake {stake} is below minimum {min_stake}")

    if stake > max_stake:
        raise InvalidStakeError(f"Stake {stake} exceeds maximum {max_stake}")


# ============================================================
# EXCEPTION HIERARCHY DOCUMENTATION
# ============================================================

"""
Exception Hierarchy:

AsianOddsError (base)
├── APIError
│   ├── APIConnectionError
│   ├── APITimeoutError
│   ├── APIRateLimitError
│   ├── APIAuthenticationError
│   ├── APIQuotaExceededError
│   ├── APIInvalidResponseError
│   └── APIKeyMissingError
├── DataValidationError
│   ├── InvalidMatchDataError
│   ├── InvalidOddsError
│   ├── InvalidProbabilityError
│   ├── InsufficientDataError
│   └── DataQualityError
├── BettingError
│   ├── InvalidStakeError
│   ├── BankrollExceededError
│   ├── RiskLimitExceededError
│   ├── InvalidMarketError
│   ├── BetPlacementError
│   └── NoValueBetError
├── AIError
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   ├── PredictionError
│   ├── CalibrationError
│   ├── ModelNotTrainedError
│   └── InsufficientTrainingDataError
├── ConfigurationError
│   ├── InvalidConfigError
│   ├── MissingConfigError
│   └── ConfigValidationError
├── DatabaseError
│   ├── DatabaseConnectionError
│   └── DatabaseQueryError
├── NotificationError
│   ├── TelegramError
│   └── TelegramRateLimitError
├── LiveBettingError
│   ├── MatchNotLiveError
│   ├── LiveDataUnavailableError
│   └── LiveDataStaleError
├── OddsTrackingError
│   ├── OddsMovementError
│   └── OddsHistoryError
└── SystemError
    ├── InitializationError
    └── ResourceError
"""
