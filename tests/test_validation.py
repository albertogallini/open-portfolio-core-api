import pytest
import pandas as pd
from oport.const_and_utils import (
    TransactionSchema, 
    PortfolioDescriptor, 
    validate_json_data, 
    validate_dataframe_schema
)

def test_portfolio_descriptor_validation():
    """Test PortfolioDescriptor validation"""
    valid_data = {"name": "Test Portfolio", "currency": "USD"}
    result = validate_json_data(valid_data, PortfolioDescriptor)
    assert result["name"] == "Test Portfolio"
    assert result["currency"] == "USD"

def test_transaction_schema_validation():
    """Test TransactionSchema validation"""
    valid_data = {"ticker": "AAPL", "quantity": 100.0, "price": 150.0}
    result = validate_json_data(valid_data, TransactionSchema)
    assert result["ticker"] == "AAPL"

def test_dataframe_validation():
    """Test DataFrame column validation"""
    df = pd.DataFrame({"date": ["2023-01-01"], "ticker": ["AAPL"], "quantity": [100]})
    expected_cols = ["date", "ticker", "quantity"]
    result = validate_dataframe_schema(df, expected_cols)
    assert not result.empty
    assert list(result.columns) == expected_cols