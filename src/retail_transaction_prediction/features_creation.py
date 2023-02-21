import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import math
import string
import numpy as np
import pyspark.sql.functions as f
import pyspark.sql.types as t
from IPython.display import display
import retail_transaction_prediction.config_parsing as cp
import utils.training_data_preparation_utils as tdpu


class FeaturesCreation:
    def __init__(self):
        self.path = cp.get_file_path()
        self.spark = (SparkSession.builder.master("local").appName("test_prediction").getOrCreate())
        self.df = None

    def read_file(self, file_format, sep=",", header=True):
        if file_format == "csv":
            self.df = self.spark.read.csv(self.path, sep=sep, header=header)
        else:
            self.df = self.spark.read.format(file_format).load(self.path)
        return self.df

    @staticmethod
    def add_features(df, buyer=True):
        df_get_purchaser_p = df.filter("Cancel_flag='P'").groupby('CustomerId') \
            .agg(f.sum("spend").alias("total_spend"), f.collect_set('Day').alias("total_buying_date"),
                 f.sum("quantity").alias("total_quantity"),
                 f.countDistinct("StockCode").alias("total_distinct_items"),
                 f.countDistinct("Invoice").alias("total_invoice"),
                 f.countDistinct("month").alias("total_distinct_month")) \
            .orderBy(f.desc('total_spend')).withColumn("delay_between_2_buying_act",
                                                       tdpu.get_buying_frequency(f.col("total_buying_date"))) \
            .withColumn("delay_in_month_between_2_buying_act",
                        f.col("delay_between_2_buying_act")/30) \
            .withColumn("total_buying_days", f.size(f.col("total_buying_date"))) \
            .select('CustomerId', "total_spend", "total_quantity", "total_distinct_items", "total_distinct_month",
                    "total_buying_days", "delay_between_2_buying_act", "delay_in_month_between_2_buying_act")

        df_get_purchaser_c = df.filter("Cancel_flag='C'").groupby('CustomerId') \
            .agg(f.sum("spend").alias("total_spend_c"),
                 f.sum("quantity").alias("total_quantity_c"),
                 f.countDistinct("StockCode").alias("total_distinct_items_c"),
                 f.countDistinct("Invoice").alias("total_invoice_c"),
                 f.countDistinct("month").alias("total_distinct_month_c")) \
            .select('CustomerId', "total_spend_c", "total_quantity_c", "total_distinct_items_c",
                    "total_distinct_month_c",
                    "total_invoice_c")
        if buyer:
            return df_get_purchaser_p.join(df_get_purchaser_c, 'CustomerId', "left").fillna(0).withColumn("label",
                                                                                                          f.lit(1))
        else:
            return df_get_purchaser_p.join(df_get_purchaser_c, 'CustomerId', "left").fillna(0).withColumn("label",
                                                                                                          f.lit(0))


    @staticmethod
    def get_data_prepared(df, month_list):
        df_prepared = tdpu.prepare_data(df, old_name="Customer ID", new_name="CustomerId")
        # tdpu.get_buying_frequency(["2011-08-05", "2011-03-25", "2011-10-21", "2009-12-15", "2010-05-10", "2010-12-07", "2010-04-22", "2011-07-28", "2011-09-28", "2011-06-09", "2009-12-21", "2010-07-23", "2009-12-11", "2011-12-09", "2010-08-09", "2010-04-27", "2011-10-03", "2010-12-09", "2010-07-16", "2011-03-03", "2010-06-23", "2010-11-10", "2011-09-22", "2011-09-15", "2010-11-15", "2011-07-04", "2010-06-08", "2009-12-01", "2010-08-26", "2010-08-11", "2010-11-08", "2011-10-04", "2010-08-17", "2009-12-03", "2010-03-05", "2010-07-30", "2010-10-14", "2011-11-04", "2011-12-08", "2011-11-28", "2010-03-29", "2011-06-07", "2010-07-15", "2010-03-18", "2010-01-12", "2010-10-17", "2010-08-31", "2010-02-25", "2009-12-22", "2011-09-02", "2010-01-05", "2010-05-21", "2010-04-12", "2010-02-19", "2011-07-20", "2010-08-02", "2011-06-14", "2010-08-01", "2010-03-30", "2010-02-05", "2011-05-17", "2011-02-07", "2011-05-16", "2010-07-27", "2010-01-08", "2010-05-17", "2011-04-20"])
        for month in month_list:
            if month == month_list[0]:
                purchaser_data = tdpu.get_purchaser_data(df_prepared, month)
                non_purchaser_data = tdpu.get_non_buyer_data(df_prepared, month)
                #purchaser_data_11_2011 = tdpu.get_purchaser_data(df_prepared, "2011-11")
                purchasers_featured = FeaturesCreation.add_features(purchaser_data)
                #purchasers_featured_11_2011 = FeaturesCreation.add_features(purchaser_data_11_2011)
                non_purchaser_featured = FeaturesCreation.add_features(non_purchaser_data, False)
                data = purchasers_featured.unionByName(non_purchaser_featured)
            else:
                purchaser_data = tdpu.get_purchaser_data(df_prepared, month)
                non_purchaser_data = tdpu.get_non_buyer_data(df_prepared, month)
                # purchaser_data_11_2011 = tdpu.get_purchaser_data(df_prepared, "2011-11")
                purchasers_featured = FeaturesCreation.add_features(purchaser_data)
                # purchasers_featured_11_2011 = FeaturesCreation.add_features(purchaser_data_11_2011)
                non_purchaser_featured = FeaturesCreation.add_features(non_purchaser_data, False)
                data = data.unionByName(purchasers_featured.unionByName(non_purchaser_featured))
        return data
