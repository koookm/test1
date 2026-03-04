"""
Sector Statistics Calculator
Computes sector-level averages for peer comparison.
"""

from typing import Dict, List
import pandas as pd
from loguru import logger


def compute_sector_stats(financials_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute sector-level average financial metrics for peer comparison.

    Args:
        financials_dict: Dict of ticker -> financials data

    Returns:
        Dict of sector -> sector average metrics
    """
    sector_data = {}

    # Group tickers by sector
    for ticker, fin_data in financials_dict.items():
        info = fin_data.get("info", {})
        sector = info.get("sector", "Unknown")

        if sector not in sector_data:
            sector_data[sector] = []

        sector_data[sector].append({
            "ticker": ticker,
            "pe_ratio": info.get("pe_ratio"),
            "pb_ratio": info.get("pb_ratio"),
            "roe": info.get("roe"),
            "roa": info.get("roa"),
            "gross_margin": info.get("gross_margin"),
            "operating_margin": info.get("operating_margin"),
            "revenue_growth": info.get("revenue_growth"),
            "debt_to_equity": info.get("debt_to_equity"),
        })

    # Compute sector averages
    sector_stats = {}
    for sector, stocks in sector_data.items():
        df = pd.DataFrame(stocks)
        stats = {}

        for col in ["pe_ratio", "pb_ratio", "roe", "roa", "gross_margin",
                    "operating_margin", "revenue_growth", "debt_to_equity"]:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    stats[f"avg_{col}"] = float(vals.mean())
                    stats[f"median_{col}"] = float(vals.median())

        sector_stats[sector] = stats

    return sector_stats


def get_peer_stats(ticker: str, sector: str, sector_stats: Dict) -> Dict:
    """Get sector average stats for a specific sector (peer comparison)."""
    return sector_stats.get(sector, {})
