"""
NYC Taxi Data Generator for Drift Detection System
Generates realistic taxi trip data with potential drift scenarios
"""

import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NYCTaxiDataGenerator:
  """Generate NYC taxi trip data with realistic patterns and drift scenarios"""
  def __init__(self): 
    self.base_date = datetime(2024, 1, 1)
    self.locations = {
            'manhattan': {'lat_range': (40.7, 40.8), 'lon_range': (-74.02, -73.9)},
            'brooklyn': {'lat_range': (40.6, 40.7), 'lon_range': (-74.05, -73.9)},
            'queens': {'lat_range': (40.65, 40.8), 'lon_range': (-73.9, -73.7)},
            'bronx': {'lat_range': (40.8, 40.9), 'lon_range': (-73.95, -73.7)},
            'staten_island': {'lat_range': (40.5, 40.65), 'lon_range': (-74.25, -74.05)}
        }
        
        # Base distributions (baseline)
    self.base_distributions = {
            'passenger_count': {'mean': 1.5, 'std': 0.8},
            'trip_duration_min': {'mean': 15, 'std': 8},
            'fare_amount': {'mean': 12.5, 'std': 6.2},
            'distance_miles': {'mean': 2.5, 'std': 1.8}
        }
        
        # Drift scenarios
    self.drift_scenarios = {
            'normal': {
                'passenger_count': {'mean': 1.5, 'std': 0.8},
                'trip_duration_min': {'mean': 15, 'std': 8},
                'fare_amount': {'mean': 12.5, 'std': 6.2},
                'distance_miles': {'mean': 2.5, 'std': 1.8}
            },
            'tourist_season': {
                'passenger_count': {'mean': 2.1, 'std': 1.2},  # More tourists
                'trip_duration_min': {'mean': 25, 'std': 12},  # Longer trips
                'fare_amount': {'mean': 18.5, 'std': 8.5},     # Higher fares
                'distance_miles': {'mean': 3.2, 'std': 2.1}
            },
            'covid_recovery': {
                'passenger_count': {'mean': 1.2, 'std': 0.6},  # Fewer passengers
                'trip_duration_min': {'mean': 12, 'std': 6},   # Shorter trips
                'fare_amount': {'mean': 10.5, 'std': 5.1},    # Lower fares
                'distance_miles': {'mean': 1.8, 'std': 1.2}
            },
            'pricing_change': {
                'passenger_count': {'mean': 1.5, 'std': 0.8},
                'trip_duration_min': {'mean': 15, 'std': 8},
                'fare_amount': {'mean': 15.8, 'std': 7.8},     # Higher pricing
                'distance_miles': {'mean': 2.5, 'std': 1.8}
            },
            'tech_issue': {
                'passenger_count': {'mean': 1.5, 'std': 0.8},
                'trip_duration_min': {'mean': 15, 'std': 8},
                'fare_amount': {'mean': 12.5, 'std': 6.2},
                'distance_miles': {'mean': 2.5, 'std': 1.8}
            }
        }
        
  def generate_trip_data(self, date: datetime, num_trips: int = 1000, 
                          scenario: str = 'normal') -> pl.DataFrame:
        """Generate taxi trip data for a specific date and scenario"""
        
        scenario_dist = self.drift_scenarios[scenario]
        trips = []
        
        for _ in range(num_trips):
            # Generate time within the day
            trip_time = date.replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            
            # Select borough with realistic weights
            borough_weights = {'manhattan': 0.4, 'brooklyn': 0.25, 'queens': 0.2, 
                             'bronx': 0.1, 'staten_island': 0.05}
            borough = np.random.choice(list(borough_weights.keys()), 
                                     p=list(borough_weights.values()))
            
            # Generate coordinates
            lat_range = self.locations[borough]['lat_range']
            lon_range = self.locations[borough]['lon_range']
            
            pickup_lat = np.random.uniform(lat_range[0], lat_range[1])
            pickup_lon = np.random.uniform(lon_range[0], lon_range[1])
            dropoff_lat = np.random.uniform(lat_range[0], lat_range[1])
            dropoff_lon = np.random.uniform(lon_range[0], lon_range[1])
            
            # Generate trip characteristics based on scenario
            passenger_count = max(1, int(np.random.normal(
                scenario_dist['passenger_count']['mean'],
                scenario_dist['passenger_count']['std']
            )))
            
            trip_duration = max(1, np.random.normal(
                scenario_dist['trip_duration_min']['mean'],
                scenario_dist['trip_duration_min']['std']
            ))
            
            fare_amount = max(2.5, np.random.normal(
                scenario_dist['fare_amount']['mean'],
                scenario_dist['fare_amount']['std']
            ))
            
            distance = max(0.1, np.random.normal(
                scenario_dist['distance_miles']['mean'],
                scenario_dist['distance_miles']['std']
            ))
            
            # Payment type
            payment_types = ['credit_card', 'cash', 'mobile']
            payment_weights = [0.7, 0.25, 0.05]
            payment_type = np.random.choice(payment_types, p=payment_weights)
            
            # Rate code
            rate_codes = [1, 2, 3, 4, 5, 6]
            rate_weights = [0.85, 0.08, 0.03, 0.02, 0.01, 0.01]
            rate_code = np.random.choice(rate_codes, p=rate_weights)
            
            trip = {
                'trip_id': f"{date.strftime('%Y%m%d')}_{_:06d}",
                'pickup_datetime': trip_time,
                'dropoff_datetime': trip_time + timedelta(minutes=trip_duration),
                'pickup_latitude': round(pickup_lat, 6),
                'pickup_longitude': round(pickup_lon, 6),
                'dropoff_latitude': round(dropoff_lat, 6),
                'dropoff_longitude': round(dropoff_lon, 6),
                'passenger_count': passenger_count,
                'trip_duration_min': round(trip_duration, 2),
                'fare_amount': round(fare_amount, 2),
                'tip_amount': round(fare_amount * np.random.uniform(0.05, 0.25), 2),
                'total_amount': round(fare_amount + fare_amount * np.random.uniform(0.05, 0.25), 2),
                'distance_miles': round(distance, 2),
                'payment_type': payment_type,
                'rate_code': rate_code,
                'borough': borough
            }
            
            trips.append(trip)
        
        df = pl.DataFrame(trips)
        
        # Introduce specific drift patterns for certain scenarios
        if scenario == 'tech_issue':
            # Introduce missing values
            missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
            df = df.with_columns([
                pl.when(pl.index().is_in(missing_indices))
                .then(None)
                .otherwise(pl.col('fare_amount'))
                .alias('fare_amount')
            ])
            
            # Introduce outliers
            outlier_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
            df = df.with_columns([
                pl.when(pl.index().is_in(outlier_indices))
                .then(pl.col('fare_amount') * 10)
                .otherwise(pl.col('fare_amount'))
                .alias('fare_amount')
            ])
        
        return df
    
  def generate_daily_batch(self, date: datetime, batch_size: int = 1000) -> pl.DataFrame:
        """Generate a daily batch of data with potential drift"""
        
        # Determine scenario based on date (simulate real drift)
        if date.month in [6, 7, 8]:  # Summer tourist season
            scenario = 'tourist_season'
        elif date.month in [3, 4]:   # Spring recovery
            scenario = 'covid_recovery'
        elif date.day % 7 == 0:      # Weekly tech issues
            scenario = 'tech_issue'
        elif date.month == 12:       # Holiday season pricing
            scenario = 'pricing_change'
        else:
            scenario = 'normal'
        
        logger.info(f"Generating data for {date.strftime('%Y-%m-%d')} with scenario: {scenario}")
        
        return self.generate_trip_data(date, batch_size, scenario)
    
  def generate_historical_data(self, start_date: datetime, end_date: datetime, 
                                batch_size: int = 1000) -> List[Tuple[datetime, pl.DataFrame]]:
        """Generate historical data for training baseline"""
        
        data_batches = []
        current_date = start_date
        
        while current_date <= end_date:
            batch = self.generate_daily_batch(current_date, batch_size)
            data_batches.append((current_date, batch))
            current_date += timedelta(days=1)
        
        return data_batches

if __name__ == "__main__":
    # Test the data generator
    generator = NYCTaxiDataGenerator()
    
    # Generate a sample batch
    sample_data = generator.generate_daily_batch(datetime(2024, 1, 15))
    print("Sample data shape:", sample_data.shape)
    print("Sample data preview:")
    print(sample_data.head())
    
    # Generate historical data for baseline
    historical_data = generator.generate_historical_data(
        datetime(2024, 1, 1), 
        datetime(2024, 1, 31), 
        batch_size=100
    )
    print(f"\nGenerated {len(historical_data)} days of historical data")