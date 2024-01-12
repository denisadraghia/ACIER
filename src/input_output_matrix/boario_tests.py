import unittest
import pandas as pd
from utils import aggregation_from_excel

class TestAggregationFromExcel(unittest.TestCase):


    def test_aggregation(self):
        # Call the function with the test data
        result = aggregation_from_excel(self.sector_excel_file, self.exio_database, self.country)

        # Add your assertions based on the expected behavior of the function
        self.assertIsInstance(result, pd.DataFrame)  # Adjust the type based on the actual return type

        # Add more specific assertions based on your function's behavior and expected results
        self.assertTrue(result.shape[0] > 0, "The aggregated database should have at least one row.")

    def test_aggregation_with_invalid_file(self):
        # Test behavior when an invalid file path is provided
        with self.assertRaises(FileNotFoundError):
            aggregation_from_excel('invalid/path/to/file.xlsx', self.exio_database, self.country)

    def test_aggregation_with_invalid_country(self):
        # Test behavior when an invalid country is provided
        with self.assertRaises(KeyError):
            aggregation_from_excel(self.sector_excel_file, self.exio_database, 'InvalidCountry')



