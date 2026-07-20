# schemas.py
# Pydantic models for validating portfolio and transaction data.

from typing import Optional
from pydantic import BaseModel


class TransactionSchema(BaseModel):
    """Schema for transaction data validation"""
    date: Optional[str] = None
    ticker: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    transaction_type: Optional[str] = None


class PortfolioDescriptor(BaseModel):
    """Schema for JSON portfolio descriptor validation"""
    name: str
    currency: Optional[str] = "USD"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark: Optional[str] = None
