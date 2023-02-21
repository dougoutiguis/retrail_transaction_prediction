import unittest
import src.retail_transaction_prediction.features_creation as fc
import src.retail_transaction_prediction.utils.training_data_preparation_utils as tdpu
import pyspark.sql.functions as F

class TestTrainingDataPreparationUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.df = fc.FeaturesCreation().read_file("csv")

    def test_rename_column(self):
        df_renamed = tdpu.rename_column(self.df.select(F.col("Customer Id")), "Customer Id", "CustomerId")

        self.assertEqual(df_renamed.columns[0], "CustomerId")

    def test_flagged_purchases(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_get_users_last_bill(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()