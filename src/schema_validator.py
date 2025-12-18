"""
Schema Validation Module using Great Expectations
Validates data quality and schema compliance for NYC taxi data
"""

import polars as pl
from datetime import datetime
import great_expectations as ge
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.data_context import DataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    DatasourceConfig,
    SimpleSqlalchemyDatasourceConfig,
)
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidator:
    """Schema validation using Great Expectations"""
    
    def __init__(self, context_path: str = "data_monitoring/gx"):
        self.context_path = context_path
        self.data_context = None
        self.expectation_suite_name = "nyc_taxi_suite"
        self._setup_context()
        
    def _setup_context(self):
        """Setup Great Expectations context and configuration"""
        try:
            # Create context directory if it doesn't exist
            os.makedirs(self.context_path, exist_ok=True)
            
            # Initialize context
            self.data_context = DataContext.create(project_root_dir=self.context_path)
            logger.info(f"Great Expectations context initialized at {self.context_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup Great Expectations context: {e}")
            raise
    
    def create_expectation_suite(self):
        """Create expectation suite for NYC taxi data"""
        try:
            # Create a new expectation suite
            suite = self.data_context.create_expectation_suite(
                expectation_suite_name=self.expectation_suite_name,
                overwrite_existing=True
            )
            
            # Add expectations for the taxi dataset
            self._add_taxi_expectations(suite)
            
            # Save the suite
            suite.save()
            logger.info(f"Expectation suite '{self.expectation_suite_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create expectation suite: {e}")
            raise
    
    def _add_taxi_expectations(self, suite):
        """Add NYC taxi specific expectations"""
        
        # Schema expectations
        suite.expect_table_columns_to_match_ordered_list([
            'trip_id', 'pickup_datetime', 'dropoff_datetime', 
            'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
            'passenger_count', 'trip_duration_min', 'fare_amount', 'tip_amount', 
            'total_amount', 'distance_miles', 'payment_type', 'rate_code', 'borough'
        ])
        
        # Column type expectations
        suite.expect_column_to_exist('trip_id')
        suite.expect_column_to_exist('pickup_datetime')
        suite.expect_column_to_exist('dropoff_datetime')
        suite.expect_column_to_exist('pickup_latitude')
        suite.expect_column_to_exist('pickup_longitude')
        suite.expect_column_to_exist('dropoff_latitude')
        suite.expect_column_to_exist('dropoff_longitude')
        suite.expect_column_to_exist('passenger_count')
        suite.expect_column_to_exist('trip_duration_min')
        suite.expect_column_to_exist('fare_amount')
        suite.expect_column_to_exist('tip_amount')
        suite.expect_column_to_exist('total_amount')
        suite.expect_column_to_exist('distance_miles')
        suite.expect_column_to_exist('payment_type')
        suite.expect_column_to_exist('rate_code')
        suite.expect_column_to_exist('borough')
        
        # Data type expectations
        suite.expect_column_values_to_be_of_type('trip_id', 'str')
        suite.expect_column_values_to_be_of_type('pickup_datetime', 'datetime64[ns]')
        suite.expect_column_values_to_be_of_type('dropoff_datetime', 'datetime64[ns]')
        suite.expect_column_values_to_be_of_type('pickup_latitude', 'float64')
        suite.expect_column_values_to_be_of_type('pickup_longitude', 'float64')
        suite.expect_column_values_to_be_of_type('dropoff_latitude', 'float64')
        suite.expect_column_values_to_be_of_type('dropoff_longitude', 'float64')
        suite.expect_column_values_to_be_of_type('passenger_count', 'int64')
        suite.expect_column_values_to_be_of_type('trip_duration_min', 'float64')
        suite.expect_column_values_to_be_of_type('fare_amount', 'float64')
        suite.expect_column_values_to_be_of_type('tip_amount', 'float64')
        suite.expect_column_values_to_be_of_type('total_amount', 'float64')
        suite.expect_column_values_to_be_of_type('distance_miles', 'float64')
        suite.expect_column_values_to_be_of_type('payment_type', 'str')
        suite.expect_column_values_to_be_of_type('rate_code', 'int64')
        suite.expect_column_values_to_be_of_type('borough', 'str')
        
        # Range expectations for coordinates
        suite.expect_column_values_to_be_between(
            'pickup_latitude', 
            min_value=40.4, max_value=41.0
        )
        suite.expect_column_values_to_be_between(
            'pickup_longitude', 
            min_value=-74.3, max_value=-73.7
        )
        suite.expect_column_values_to_be_between(
            'dropoff_latitude', 
            min_value=40.4, max_value=41.0
        )
        suite.expect_column_values_to_be_between(
            'dropoff_longitude', 
            min_value=-74.3, max_value=-73.7
        )
        
        # Business logic expectations
        suite.expect_column_values_to_be_between(
            'passenger_count', 
            min_value=1, max_value=8
        )
        suite.expect_column_values_to_be_between(
            'trip_duration_min', 
            min_value=1, max_value=180
        )
        suite.expect_column_values_to_be_between(
            'fare_amount', 
            min_value=2.5, max_value=200
        )
        suite.expect_column_values_to_be_between(
            'distance_miles', 
            min_value=0.1, max_value=50
        )
        
        # Categorical expectations
        suite.expect_column_values_to_be_in_set(
            'payment_type', 
            ['credit_card', 'cash', 'mobile']
        )
        suite.expect_column_values_to_be_in_set(
            'rate_code', 
            [1, 2, 3, 4, 5, 6]
        )
        suite.expect_column_values_to_be_in_set(
            'borough', 
            ['manhattan', 'brooklyn', 'queens', 'bronx', 'staten_island']
        )
        
        # Missing value expectations
        suite.expect_column_values_to_not_be_null('trip_id', mostly=0.99)
        suite.expect_column_values_to_not_be_null('pickup_datetime', mostly=0.99)
        suite.expect_column_values_to_not_be_null('dropoff_datetime', mostly=0.99)
        suite.expect_column_values_to_not_be_null('pickup_latitude', mostly=0.99)
        suite.expect_column_values_to_not_be_null('pickup_longitude', mostly=0.99)
        suite.expect_column_values_to_not_be_null('dropoff_latitude', mostly=0.99)
        suite.expect_column_values_to_not_be_null('dropoff_longitude', mostly=0.99)
        suite.expect_column_values_to_not_be_null('passenger_count', mostly=0.99)
        suite.expect_column_values_to_not_be_null('trip_duration_min', mostly=0.99)
        suite.expect_column_values_to_not_be_null('fare_amount', mostly=0.95)  # Allow some nulls
        
        # Logical consistency expectations
        suite.expect_column_values_to_be_greater_than(
            'dropoff_datetime', 
            'pickup_datetime'
        )
        suite.expect_column_values_to_be_greater_than(
            'total_amount', 
            'fare_amount'
        )
    
    def validate_data(self, data: pl.DataFrame, batch_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate data using Great Expectations"""
        
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Convert Polars to pandas for Great Expectations compatibility
            df_pandas = data.to_pandas()
            
            # Create data context
            context = self.data_context
            
            # Create a batch
            batch_kwargs = {
                "datasource": "my_datasource",
                "path": f"temp_batch_{batch_id}.csv",
                "batch_id": batch_id
            }
            
            # For testing, we'll use in-memory data
            batch = context.get_batch(
                batch_kwargs=batch_kwargs,
                expectation_suite_name=self.expectation_suite_name,
                batch_data=df_pandas
            )
            
            # Validate the batch
            results = context.run_checkpoint(
                checkpoint_name="my_checkpoint",
                validations=[
                    {
                        "batch_request": {
                            "batch_kwargs": batch_kwargs
                        },
                        "expectation_suite_name": self.expectation_suite_name,
                    }
                ],
                run_name=batch_id
            )
            
            # Convert results to our format
            validation_result = self._process_validation_results(results, batch_id)
            
            logger.info(f"Validation completed for batch {batch_id}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed for batch {batch_id}: {e}")
            return {
                'batch_id': batch_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_validation_results(self, results, batch_id: str) -> Dict[str, Any]:
        """Process Great Expectations validation results"""
        
        try:
            # Extract results from the validation
            validation_results = []
            total_checks = 0
            passed_checks = 0
            
            if hasattr(results, 'list_validation_results'):
                for result in results.list_validation_results():
                    if hasattr(result, 'success'):
                        validation_results.append({
                            'success': result.success,
                            'result': result,
                            'expectation_type': getattr(result, 'expectation_type', 'unknown')
                        })
                        
                        total_checks += 1
                        if result.success:
                            passed_checks += 1
            
            success_rate = (passed_checks / total_checks) if total_checks > 0 else 0
            
            return {
                'batch_id': batch_id,
                'success': success_rate > 0.8,  # 80% threshold
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'success_rate': success_rate,
                'results': validation_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process validation results: {e}")
            return {
                'batch_id': batch_id,
                'success': False,
                'error': f"Failed to process results: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation results"""
        
        if not validation_result.get('success', False):
            return {
                'status': 'FAIL',
                'summary': 'Validation failed',
                'details': validation_result
            }
        
        success_rate = validation_result.get('success_rate', 0)
        
        if success_rate > 0.95:
            status = 'EXCELLENT'
        elif success_rate > 0.90:
            status = 'GOOD'
        elif success_rate > 0.80:
            status = 'ACCEPTABLE'
        else:
            status = 'POOR'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'total_checks': validation_result.get('total_checks', 0),
            'passed_checks': validation_result.get('passed_checks', 0),
            'timestamp': validation_result.get('timestamp')
        }

if __name__ == "__main__":
    # Test the schema validator
    validator = SchemaValidator()
    
    # Create expectation suite
    validator.create_expectation_suite()
    
    # Create sample data for testing    
    sample_data = pl.DataFrame({
        'trip_id': ['trip_001', 'trip_002'],
        'pickup_datetime': [datetime.now(), datetime.now()],
        'dropoff_datetime': [datetime.now(), datetime.now()],
        'pickup_latitude': [40.7589, 40.7505],
        'pickup_longitude': [-73.9851, -73.9934],
        'dropoff_latitude': [40.7505, 40.7589],
        'dropoff_longitude': [-73.9934, -73.9851],
        'passenger_count': [1, 2],
        'trip_duration_min': [15.5, 20.0],
        'fare_amount': [12.50, 18.75],
        'tip_amount': [2.50, 3.75],
        'total_amount': [15.00, 22.50],
        'distance_miles': [2.5, 3.2],
        'payment_type': ['credit_card', 'cash'],
        'rate_code': [1, 1],
        'borough': ['manhattan', 'manhattan']
    })
    
    # Test validation
    result = validator.validate_data(sample_data)
    print("Validation result:", result)
    
    summary = validator.get_validation_summary(result)
    print("Validation summary:", summary)