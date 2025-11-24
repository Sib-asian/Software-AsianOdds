# Error Handling Best Practices

This document outlines best practices for error handling in the AsianOdds betting system.

## General Principles

1. **Use Specific Exceptions**: Always use the most specific exception type available from `exceptions.py`
2. **Never Silence Errors**: Avoid bare `except: pass` statements
3. **Log All Errors**: Always log exceptions with appropriate context
4. **Fail Fast**: Detect and raise errors early rather than propagating invalid state
5. **Provide Context**: Include helpful error messages with details about what went wrong

## Using Custom Exceptions

Import exceptions from the centralized `exceptions.py` module:

```python
from exceptions import (
    APIError, APIRateLimitError,
    DataValidationError, InvalidOddsError,
    BettingError, InvalidStakeError
)
```

## Exception Handling Patterns

### ❌ BAD: Generic Exception Catching

```python
# DON'T DO THIS
try:
    result = api.fetch_match_data(match_id)
except Exception as e:
    logger.error(f"Error: {e}")
    pass  # Silently ignoring error
```

**Problems:**
- Catches all exceptions, even ones you don't expect
- Loses information about error type
- Silent failure hides bugs

### ✅ GOOD: Specific Exception Handling

```python
# DO THIS INSTEAD
from exceptions import APIError, APIRateLimitError, InvalidMatchDataError

try:
    result = api.fetch_match_data(match_id)
except APIRateLimitError as e:
    logger.warning(f"Rate limit hit for match {match_id}, retrying after {e.retry_after}s")
    time.sleep(e.retry_after)
    return None
except APIError as e:
    logger.error(f"API error fetching match {match_id}: {e}")
    raise  # Re-raise to let caller handle it
except InvalidMatchDataError as e:
    logger.error(f"Invalid data for match {match_id}: {e}")
    return None
```

**Benefits:**
- Handles different errors appropriately
- Clear intent for each error type
- Proper logging and recovery

## Common Patterns

### 1. API Calls with Retry Logic

```python
from exceptions import APIError, APIRateLimitError, APITimeoutError

def fetch_with_retry(url: str, max_retries: int = 3) -> dict:
    """Fetch data with automatic retry on transient errors"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except APIRateLimitError as e:
            if e.retry_after:
                logger.warning(f"Rate limited, waiting {e.retry_after}s")
                time.sleep(e.retry_after)
            else:
                time.sleep(2 ** attempt)  # Exponential backoff

        except APITimeoutError:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise

        except APIError as e:
            logger.error(f"API error: {e}")
            raise  # Don't retry on non-transient errors

    raise APIError(f"Failed after {max_retries} attempts")
```

### 2. Data Validation

```python
from exceptions import InvalidOddsError, InvalidProbabilityError, validate_odds, validate_probability

def calculate_ev(probability: float, odds: float) -> float:
    """Calculate expected value with validation"""
    # Use validation helpers
    validate_probability(probability, "win probability")
    validate_odds(odds, "decimal odds")

    # Calculate EV
    ev = (probability * odds) - 1
    return ev
```

### 3. Database Operations

```python
from database_manager import DatabaseManager
from exceptions import DatabaseError, DatabaseQueryError

def save_prediction(db: DatabaseManager, prediction: dict):
    """Save prediction with proper error handling"""
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO predictions (match_id, prob, confidence) VALUES (?, ?, ?)",
                (prediction['match_id'], prediction['prob'], prediction['confidence'])
            )
            logger.info(f"✅ Saved prediction for match {prediction['match_id']}")

    except DatabaseQueryError as e:
        logger.error(f"Failed to save prediction: {e}")
        raise  # Let caller decide how to handle

    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise
```

### 4. Model Predictions

```python
from exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    PredictionError,
    InsufficientDataError
)

def predict_match_outcome(model, match_data: dict) -> dict:
    """Make prediction with comprehensive error handling"""
    # Validate input
    if not match_data or 'teams' not in match_data:
        raise InsufficientDataError("Match data incomplete")

    try:
        # Load model if needed
        if not model.is_loaded:
            raise ModelNotFoundError("Model not loaded")

        # Make prediction
        prediction = model.predict(match_data)

        # Validate output
        if prediction is None or 'probability' not in prediction:
            raise PredictionError("Model returned invalid prediction")

        return prediction

    except ModelLoadError as e:
        logger.error(f"Failed to load model: {e}")
        raise

    except PredictionError as e:
        logger.error(f"Prediction failed: {e}")
        raise

    except Exception as e:
        # Unexpected error - wrap in PredictionError
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise PredictionError(f"Unexpected prediction error: {e}") from e
```

### 5. Configuration Validation

```python
from exceptions import ConfigurationError, MissingConfigError, InvalidConfigError

def load_config(config_path: str) -> dict:
    """Load and validate configuration"""
    if not Path(config_path).exists():
        raise MissingConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = json.load(f)

    except json.JSONDecodeError as e:
        raise InvalidConfigError(f"Invalid JSON in config: {e}")

    # Validate required fields
    required_fields = ['api_key', 'min_confidence', 'min_ev']
    missing = [f for f in required_fields if f not in config]

    if missing:
        raise InvalidConfigError(f"Missing required config fields: {missing}")

    # Validate ranges
    if not (0 < config['min_confidence'] < 100):
        raise InvalidConfigError(f"min_confidence must be 0-100, got {config['min_confidence']}")

    return config
```

## Error Logging

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: Confirmation that things are working
- **WARNING**: Something unexpected but handled
- **ERROR**: Serious problem that prevented operation
- **CRITICAL**: System-level failure

### Logging Examples

```python
# DEBUG - Trace execution
logger.debug(f"Fetching data for match {match_id}")

# INFO - Success message
logger.info(f"✅ Successfully processed {count} matches")

# WARNING - Handled error
logger.warning(f"⚠️  Low data quality ({quality:.1f}%), using fallback")

# ERROR - Operation failed
logger.error(f"❌ Failed to save prediction: {e}")

# ERROR with traceback
logger.error(f"❌ Unexpected error", exc_info=True)

# CRITICAL - System failure
logger.critical(f"🚨 Database connection lost, shutting down")
```

## Converting Legacy Code

When refactoring old code with generic exceptions:

### Before (Bad)
```python
try:
    odds = fetch_odds(match_id)
    if odds < 1.0:
        return None
except Exception:
    return None
```

### After (Good)
```python
from exceptions import APIError, InvalidOddsError, validate_odds

try:
    odds = fetch_odds(match_id)
    validate_odds(odds)
    return odds
except APIError as e:
    logger.error(f"Failed to fetch odds for match {match_id}: {e}")
    return None
except InvalidOddsError as e:
    logger.warning(f"Invalid odds for match {match_id}: {e}")
    return None
```

## Exception Documentation

All custom exceptions are documented in `exceptions.py`. See the exception hierarchy at the bottom of that file.

## Testing Error Handling

Always test error paths:

```python
import pytest
from exceptions import InvalidOddsError

def test_validate_odds_rejects_invalid():
    """Test that invalid odds raise appropriate exception"""
    with pytest.raises(InvalidOddsError):
        validate_odds(0.5)  # Odds must be > 1.0

    with pytest.raises(InvalidOddsError):
        validate_odds(-1.0)  # Negative odds

def test_api_handles_rate_limit():
    """Test that rate limit is handled gracefully"""
    # Mock API to raise rate limit
    with mock.patch('api.fetch', side_effect=APIRateLimitError("Rate limited", retry_after=60)):
        result = fetch_with_retry("http://example.com")
        assert result is None  # Should handle gracefully
```

## Summary

✅ **DO:**
- Use specific exception types
- Log all errors with context
- Validate inputs early
- Provide helpful error messages
- Test error handling paths
- Re-raise when appropriate

❌ **DON'T:**
- Use bare `except Exception`
- Silently catch and ignore errors
- Use `pass` after catching
- Let invalid data propagate
- Forget to log errors
- Catch exceptions you can't handle

## Resources

- Custom exceptions: `exceptions.py`
- Database manager: `database_manager.py`
- Configuration: `ai_system/config.py`
- Validation helpers: `exceptions.py` (validate_* functions)
