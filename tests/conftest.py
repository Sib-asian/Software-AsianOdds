"""
Pytest configuration and shared fixtures
=========================================

Shared test fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path to tests"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Provide test data directory path"""
    return project_root_path / "tests" / "test_data"


@pytest.fixture
def sample_match_data():
    """Provide sample match data for testing"""
    return {
        "fixture": {
            "id": 12345,
            "date": "2025-11-24T20:00:00Z",
            "status": {"short": "NS"}
        },
        "teams": {
            "home": {"id": 1, "name": "Team A"},
            "away": {"id": 2, "name": "Team B"}
        },
        "league": {
            "id": 100,
            "name": "Test League",
            "country": "Italy"
        }
    }


@pytest.fixture
def sample_odds_data():
    """Provide sample odds data for testing"""
    return {
        "home": 2.50,
        "draw": 3.20,
        "away": 2.80
    }


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration before each test"""
    # This runs before each test to ensure clean state
    yield
    # Cleanup after test if needed
