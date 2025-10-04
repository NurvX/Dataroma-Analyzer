#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataroma Investment Analyzer - Position Timeline Analyzer

Tracks position building and reduction over time for manager-stock combinations.
Shows accumulation vs distribution phases, position sizing progression,
and identifies when managers flip from building to reducing positions.

MIT License
Copyright (c) 2020-present Jerzy 'Yuri' Kramarz
See LICENSE file for full license text.

Author: Jerzy 'Yuri' Kramarz
Source: https://github.com/op7ic/Dataroma-Analyzer
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re

from .base_analyzer import BaseAnalyzer, MultiAnalyzer
from ..data.data_loader import DataLoader


class PositionTimelineAnalyzer(MultiAnalyzer):
    """Analyzes position building and reduction patterns over time."""

    def __init__(self, data_loader: DataLoader) -> None:
        """Initialize with data loader."""
        super().__init__(data_loader)

    def analyze_all(self) -> Dict[str, pd.DataFrame]:
        """Run all position timeline analyses."""
        results = {}

        results["position_building_timeline"] = self.analyze_position_building_timeline()
        results["accumulation_vs_distribution"] = self.analyze_accumulation_distribution_phases()
        results["position_flip_points"] = self.analyze_position_flip_points()

        for name, df in results.items():
            self.log_analysis_summary(df, name)

        return self.format_all_outputs(results)

    def analyze_position_building_timeline(self) -> pd.DataFrame:
        """
        Create detailed timeline showing how managers build/reduce positions over time.

        Shows quarter-by-quarter progression of positions with cumulative tracking.

        Returns:
            DataFrame with position timeline for significant manager-stock combinations
        """
        if self.data.history_df is None or self.data.history_df.empty:
            return pd.DataFrame()

        print("📈 Analyzing Position Building Timelines...")

        # Get all manager-stock combinations with 5+ activities (significant positions)
        position_counts = self.data.history_df.groupby(['ticker', 'manager_id']).size()
        significant_positions = position_counts[position_counts >= 5].index.tolist()

        timeline_data = []

        for ticker, manager in significant_positions:
            # Get all activities for this manager-stock combination
            activities = self.data.history_df[
                (self.data.history_df['ticker'] == ticker) &
                (self.data.history_df['manager_id'] == manager)
            ].copy()

            if activities.empty:
                continue

            # Sort by period chronologically
            activities = activities.sort_values('period')

            # Track cumulative shares over time
            cumulative_shares = 0
            previous_shares = 0

            for _, act in activities.iterrows():
                action_type = act.get('action_type', 'Hold')
                shares = act.get('shares', 0)
                action_str = act.get('action', '')
                quarter = act.get('period', '')

                # Update cumulative shares based on action type
                if action_type == 'Buy':
                    # New position
                    cumulative_shares = shares
                elif action_type == 'Add':
                    # Parse Add percentage if available
                    match = re.search(r'(\d+\.?\d*)%', action_str)
                    if match:
                        add_pct = float(match.group(1)) / 100
                        # shares shown = shares after add
                        # If added X%, then: shares_after = shares_before * (1 + X/100)
                        # So: shares_before = shares / (1 + X/100)
                        shares_before = shares / (1 + add_pct) if add_pct > 0 else shares
                        shares_added = shares - shares_before
                        cumulative_shares = shares
                    else:
                        # No percentage, assume shares is new total
                        cumulative_shares = shares
                elif action_type == 'Reduce':
                    # shares shown = shares after reduction
                    cumulative_shares = shares
                elif action_type == 'Sell':
                    # Position sold
                    if '100.00%' in action_str or '100%' in action_str:
                        cumulative_shares = 0
                    else:
                        # Partial sell
                        cumulative_shares = shares
                else:
                    # Hold - no change
                    cumulative_shares = previous_shares

                # Determine phase (last 3 actions)
                recent_actions = activities[activities['period'] <= quarter].tail(3)['action_type'].tolist()

                if len(recent_actions) >= 2:
                    buy_add_count = sum(1 for a in recent_actions if a in ['Buy', 'Add'])
                    sell_reduce_count = sum(1 for a in recent_actions if a in ['Sell', 'Reduce'])

                    if buy_add_count > sell_reduce_count:
                        phase = 'Accumulating'
                    elif sell_reduce_count > buy_add_count:
                        phase = 'Distributing'
                    else:
                        phase = 'Transitioning'
                else:
                    phase = 'Initial'

                timeline_data.append({
                    'ticker': ticker,
                    'manager_id': manager,
                    'manager_name': self.data.manager_names.get(manager, manager),
                    'quarter': quarter,
                    'action_type': action_type,
                    'action': action_str,
                    'shares': shares,
                    'cumulative_shares': cumulative_shares,
                    'phase': phase,
                    'quarter_change': cumulative_shares - previous_shares if previous_shares > 0 else cumulative_shares
                })

                previous_shares = cumulative_shares

        if not timeline_data:
            return pd.DataFrame()

        timeline_df = pd.DataFrame(timeline_data)

        # Add company names
        if self.data.holdings_df is not None and 'stock' in self.data.holdings_df.columns:
            company_names = self.data.holdings_df.groupby('ticker')['stock'].first()
            timeline_df = timeline_df.merge(
                company_names.to_frame('company_name'),
                left_on='ticker',
                right_index=True,
                how='left'
            )

        # Sort by ticker, manager, and period
        timeline_df = timeline_df.sort_values(['ticker', 'manager_id', 'quarter'])

        return self.format_output(timeline_df)

    def analyze_accumulation_distribution_phases(self) -> pd.DataFrame:
        """
        Identify current phase (accumulating vs distributing) for each manager-stock position.

        Returns:
            DataFrame showing which positions are being built up vs reduced
        """
        if self.data.history_df is None or self.data.history_df.empty:
            return pd.DataFrame()

        print("🔄 Analyzing Accumulation vs Distribution Phases...")

        # Get recent 4 quarters to determine current trend
        recent_quarters = self.get_recent_quarters(4)

        phase_analysis = []

        # Get current holdings
        if self.data.holdings_df is not None and not self.data.holdings_df.empty:
            for _, holding in self.data.holdings_df.iterrows():
                ticker = holding['ticker']
                manager = holding['manager_id']

                # Get recent activities
                recent_activities = self.data.history_df[
                    (self.data.history_df['ticker'] == ticker) &
                    (self.data.history_df['manager_id'] == manager) &
                    (self.data.history_df['period'].isin(recent_quarters))
                ]

                if recent_activities.empty:
                    continue

                # Count action types
                action_counts = recent_activities['action_type'].value_counts().to_dict()
                buy_add = action_counts.get('Buy', 0) + action_counts.get('Add', 0)
                sell_reduce = action_counts.get('Sell', 0) + action_counts.get('Reduce', 0)

                # Determine phase
                if buy_add > sell_reduce:
                    phase = 'Accumulating'
                    conviction = 'Building'
                elif sell_reduce > buy_add:
                    phase = 'Distributing'
                    conviction = 'Reducing'
                else:
                    phase = 'Mixed'
                    conviction = 'Uncertain'

                # Get most recent action
                most_recent = recent_activities.sort_values('period', ascending=False).iloc[0]

                phase_analysis.append({
                    'ticker': ticker,
                    'company_name': holding.get('stock', ''),
                    'manager_id': manager,
                    'manager_name': self.data.manager_names.get(manager, manager),
                    'current_shares': holding.get('shares', 0),
                    'current_value': holding.get('value', 0),
                    'portfolio_percent': holding.get('portfolio_percent', 0),
                    'recent_quarters': len(recent_activities),
                    'buy_add_actions': buy_add,
                    'sell_reduce_actions': sell_reduce,
                    'net_activity': buy_add - sell_reduce,
                    'phase': phase,
                    'conviction_trend': conviction,
                    'last_action': most_recent.get('action', ''),
                    'last_quarter': most_recent.get('period', '')
                })

        if not phase_analysis:
            return pd.DataFrame()

        phase_df = pd.DataFrame(phase_analysis)

        # Sort by phase and net activity
        phase_df = phase_df.sort_values(['phase', 'net_activity'], ascending=[True, False])

        return self.format_output(phase_df).head(200)

    def analyze_position_flip_points(self) -> pd.DataFrame:
        """
        Identify when managers switched from accumulation to distribution.

        These "flip points" can signal important changes in conviction.

        Returns:
            DataFrame with position flip points
        """
        if self.data.history_df is None or self.data.history_df.empty:
            return pd.DataFrame()

        print("🔄 Identifying Position Flip Points...")

        flip_points = []

        # Group by ticker and manager
        for (ticker, manager), group in self.data.history_df.groupby(['ticker', 'manager_id']):
            if len(group) < 4:
                continue

            # Sort chronologically
            group = group.sort_values('period')

            # Look for transitions from accumulation to distribution
            actions = group['action_type'].tolist()
            periods = group['period'].tolist()

            for i in range(len(actions) - 2):
                # Get 3-action window
                window = actions[i:i+3]

                # Check for flip: 2+ accumulation actions → 2+ distribution actions
                early_accum = sum(1 for a in window[:2] if a in ['Buy', 'Add'])
                late_distrib = sum(1 for a in window[1:] if a in ['Sell', 'Reduce'])

                if early_accum >= 2 and late_distrib >= 1 and window[-1] in ['Sell', 'Reduce']:
                    # Found a flip point
                    flip_quarter = periods[i+2]

                    # Get shares at flip point
                    flip_action = group.iloc[i+2]

                    flip_points.append({
                        'ticker': ticker,
                        'manager_id': manager,
                        'manager_name': self.data.manager_names.get(manager, manager),
                        'flip_quarter': flip_quarter,
                        'flip_action': flip_action.get('action', ''),
                        'shares_at_flip': flip_action.get('shares', 0),
                        'action_sequence': ' → '.join(window),
                        'quarters_before_flip': i + 1,
                        'quarters_after_flip': len(actions) - (i + 3)
                    })

        if not flip_points:
            return pd.DataFrame()

        flip_df = pd.DataFrame(flip_points)

        # Add company names
        if self.data.holdings_df is not None and 'stock' in self.data.holdings_df.columns:
            company_names = self.data.holdings_df.groupby('ticker')['stock'].first()
            flip_df = flip_df.merge(
                company_names.to_frame('company_name'),
                left_on='ticker',
                right_index=True,
                how='left'
            )

        # Add current status
        if self.data.holdings_df is not None:
            current_holdings = self.data.holdings_df.groupby(['ticker', 'manager_id']).agg({
                'shares': 'first',
                'value': 'first'
            })
            flip_df = flip_df.merge(
                current_holdings.add_prefix('current_'),
                left_on=['ticker', 'manager_id'],
                right_index=True,
                how='left'
            )
            flip_df['still_held'] = flip_df['current_shares'].notna()
        else:
            flip_df['still_held'] = False

        # Sort by flip quarter (most recent first)
        flip_df = flip_df.sort_values('flip_quarter', ascending=False)

        return self.format_output(flip_df).head(100)
