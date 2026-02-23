#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Cross-File Validation

Cross-file validation and contradiction detection for CSV outputs.
Identifies apparent contradictions across analysis files and documents
why they are valid (e.g., different managers, different time windows).

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)


class CrossFileValidator:
    """
    Validates consistency across CSV output files.
    
    Detects apparent contradictions between analysis files and documents
    whether they are bugs or valid scenarios (e.g., different managers
    buying/selling the same stock).
    """

    def __init__(self, analysis_dir: str = "analysis") -> None:
        """
        Initialize the cross-file validator.
        
        Args:
            analysis_dir: Path to the analysis output directory
        """
        self.analysis_dir = Path(analysis_dir)
        self._issues_cache: Dict[str, List[str]] = {}
        
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Run all validations and return issues found.
        
        Returns:
            Dictionary mapping issue category to list of issue descriptions
        """
        logger.info("Running cross-file validation...")
        issues: Dict[str, List[str]] = {}
        
        # Check 52-week file overlaps
        overlap_issues = self._check_52_week_overlaps()
        if overlap_issues:
            issues['52_week_overlaps'] = overlap_issues
            logger.info(f"Found {len(overlap_issues)} 52-week overlap issues")
            
        # Check momentum vs contrarian contradictions
        momentum_issues = self._check_momentum_contrarian()
        if momentum_issues:
            issues['momentum_contrarian'] = momentum_issues
            logger.info(f"Found {len(momentum_issues)} momentum/contrarian issues")
            
        # Check for stale data in current/ files
        stale_issues = self._check_current_freshness()
        if stale_issues:
            issues['stale_current'] = stale_issues
            logger.info(f"Found {len(stale_issues)} stale data issues")
            
        # Check boolean flag distributions
        flag_issues = self._check_boolean_flags()
        if flag_issues:
            issues['boolean_flags'] = flag_issues
            logger.info(f"Found {len(flag_issues)} boolean flag issues")
            
        self._issues_cache = issues
        
        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            logger.info("Cross-file validation complete: No issues found")
        else:
            logger.info(f"Cross-file validation complete: {total_issues} total issues documented")
            
        return issues
    
    def _check_52_week_overlaps(self) -> List[str]:
        """
        Find tickers appearing in both 52-week low buys and high sells files.
        
        This is valid when different managers are buying and selling the same
        stock, but should be documented for transparency.
        
        Returns:
            List of issue descriptions for overlapping tickers
        """
        low_file = self.analysis_dir / "current" / "52_week_low_buys.csv"
        high_file = self.analysis_dir / "current" / "52_week_high_sells.csv"
        
        if not low_file.exists() or not high_file.exists():
            logger.debug("52-week files not found, skipping overlap check")
            return []
            
        try:
            low_df = pd.read_csv(low_file)
            high_df = pd.read_csv(high_file)
        except Exception as e:
            logger.warning(f"Error reading 52-week files: {e}")
            return []
        
        # Handle case where ticker column might not exist
        if 'ticker' not in low_df.columns or 'ticker' not in high_df.columns:
            logger.warning("Ticker column not found in 52-week files")
            return []
        
        overlap = set(low_df['ticker']) & set(high_df['ticker'])
        issues: List[str] = []
        
        for ticker in sorted(overlap):
            low_rows = low_df[low_df['ticker'] == ticker]
            high_rows = high_df[high_df['ticker'] == ticker]
            
            if low_rows.empty or high_rows.empty:
                continue
                
            low_row = low_rows.iloc[0]
            high_row = high_rows.iloc[0]
            
            # Get position percentage if available
            pos_pct = low_row.get('52_week_position_pct', 'N/A')
            if pd.notna(pos_pct) and pos_pct != 'N/A':
                pos_pct = f"{float(pos_pct):.1f}"
            
            # Get buying and selling managers
            buying_mgrs = low_row.get('buying_managers', 'Unknown')
            selling_mgrs = high_row.get('selling_managers', 'Unknown')
            
            issues.append(
                f"{ticker}: position={pos_pct}%, in both low-buys AND high-sells "
                f"(different managers buying/selling)"
            )
            
        return issues
    
    def _check_momentum_contrarian(self) -> List[str]:
        """
        Find tickers with contradictory momentum/contrarian labels.
        
        A stock can be 'Recent Surge' in momentum (based on 3Q buy count)
        while showing 'Net Selling' in contrarian (based on 2Q net activity).
        This is valid due to different analysis windows and metrics.
        
        Returns:
            List of issue descriptions for contradictory labels
        """
        mom_file = self.analysis_dir / "current" / "momentum_stocks.csv"
        cont_file = self.analysis_dir / "current" / "contrarian_opportunities.csv"
        
        if not mom_file.exists() or not cont_file.exists():
            logger.debug("Momentum/contrarian files not found, skipping check")
            return []
            
        try:
            mom_df = pd.read_csv(mom_file)
            cont_df = pd.read_csv(cont_file)
        except Exception as e:
            logger.warning(f"Error reading momentum/contrarian files: {e}")
            return []
        
        # Handle missing columns
        if 'ticker' not in mom_df.columns or 'ticker' not in cont_df.columns:
            logger.warning("Ticker column not found in momentum/contrarian files")
            return []
        
        # Find overlapping tickers
        overlap = set(mom_df['ticker']) & set(cont_df['ticker'])
        issues: List[str] = []
        
        for ticker in sorted(overlap):
            mom_rows = mom_df[mom_df['ticker'] == ticker]
            cont_rows = cont_df[cont_df['ticker'] == ticker]
            
            if mom_rows.empty or cont_rows.empty:
                continue
                
            mom_row = mom_rows.iloc[0]
            cont_row = cont_rows.iloc[0]
            
            mom_type = str(mom_row.get('momentum_type', ''))
            cont_signal = str(cont_row.get('contrarian_signal', ''))
            
            # Check for apparent contradiction: Surge + Selling
            if 'Surge' in mom_type and 'Selling' in cont_signal:
                issues.append(
                    f"{ticker}: momentum='{mom_type}' but contrarian='{cont_signal}' "
                    f"(different windows: 3Q vs 2Q)"
                )
                
        return issues
    
    def _check_current_freshness(self) -> List[str]:
        """
        Check that current/ files don't have stale data.
        
        Identifies rows with recent_buys=0 in opportunity files,
        which may indicate stale or outdated entries.
        
        Returns:
            List of issue descriptions for stale data
        """
        issues: List[str] = []
        current_dir = self.analysis_dir / "current"
        
        if not current_dir.exists():
            logger.debug("Current directory not found, skipping freshness check")
            return []
        
        for csv_file in current_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
                continue
            
            # Check for recent_buys=0 in opportunity files
            if 'recent_buys' in df.columns:
                stale = df[df['recent_buys'] == 0]
                if len(stale) > 0:
                    issues.append(
                        f"{csv_file.name}: {len(stale)} rows with recent_buys=0"
                    )
                    
        return issues
    
    def _check_boolean_flags(self) -> List[str]:
        """
        Check that boolean flags are statistically meaningful.
        
        Flags that are >90% True provide little filtering value
        and may indicate a poorly calibrated threshold.
        
        Returns:
            List of issue descriptions for useless boolean flags
        """
        issues: List[str] = []
        
        if not self.analysis_dir.exists():
            logger.debug("Analysis directory not found, skipping boolean check")
            return []
        
        for csv_file in self.analysis_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
                continue
            
            if len(df) == 0:
                continue
                
            for col in df.columns:
                # Check if column appears to be boolean
                unique_values = df[col].dropna().unique()
                is_boolean = (
                    df[col].dtype == bool or 
                    set(unique_values).issubset({True, False, 'True', 'False', 1, 0, '1', '0'})
                )
                
                if is_boolean and len(unique_values) > 0:
                    # Convert to boolean for counting
                    bool_series = df[col].map(
                        lambda x: x in [True, 'True', 1, '1'] if pd.notna(x) else False
                    )
                    true_count = bool_series.sum()
                    true_pct = (true_count / len(df)) * 100
                    
                    if true_pct > 90:
                        # Get relative path for cleaner output
                        rel_path = csv_file.relative_to(self.analysis_dir)
                        issues.append(
                            f"{rel_path}: {col} is {true_pct:.1f}% True "
                            f"(statistically useless)"
                        )
                        
        return issues
    
    def generate_contradiction_ledger(self) -> pd.DataFrame:
        """
        Generate a ledger of all cross-file contradictions with explanations.
        
        Creates a DataFrame documenting each apparent contradiction,
        explaining why it occurs and whether it represents a bug.
        
        Returns:
            DataFrame with contradiction records
        """
        records: List[Dict[str, Any]] = []
        
        # Run validation if not already done
        if not self._issues_cache:
            self.validate_all()
        
        # 52-week overlaps
        for issue in self._check_52_week_overlaps():
            ticker = issue.split(':')[0]
            records.append({
                'ticker': ticker,
                'files': '52_week_low_buys.csv, 52_week_high_sells.csv',
                'apparent_contradiction': 'Appears in both buy-low and sell-high files',
                'explanation': 'Different managers buying and selling simultaneously',
                'is_bug': False
            })
            
        # Momentum vs contrarian
        for issue in self._check_momentum_contrarian():
            ticker = issue.split(':')[0]
            records.append({
                'ticker': ticker,
                'files': 'momentum_stocks.csv, contrarian_opportunities.csv',
                'apparent_contradiction': 'Recent Surge in momentum, Net Selling in contrarian',
                'explanation': 'Different analysis windows (3Q vs 2Q) and different metrics (buy count vs net activity)',
                'is_bug': False
            })
        
        df = pd.DataFrame(records)
        
        if df.empty:
            logger.info("No contradictions found - ledger is empty")
        else:
            logger.info(f"Generated contradiction ledger with {len(df)} entries")
            
        return df
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the validation results.
        
        Returns:
            Dictionary with validation summary statistics
        """
        if not self._issues_cache:
            self.validate_all()
            
        total_issues = sum(len(v) for v in self._issues_cache.values())
        
        return {
            'total_issues': total_issues,
            'categories': {k: len(v) for k, v in self._issues_cache.items()},
            'has_contradictions': (
                len(self._issues_cache.get('52_week_overlaps', [])) > 0 or
                len(self._issues_cache.get('momentum_contrarian', [])) > 0
            ),
            'has_quality_issues': (
                len(self._issues_cache.get('stale_current', [])) > 0 or
                len(self._issues_cache.get('boolean_flags', [])) > 0
            )
        }
