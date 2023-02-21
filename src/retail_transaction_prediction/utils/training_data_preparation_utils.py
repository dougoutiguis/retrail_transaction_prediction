import pyspark.sql.functions as f
import pyspark.sql.types as t


def get_bill_period(df, begin, end):
    return df.filter(f.col("month").between(begin, end))


def rename_column(df, old_name, new_name):
    cols = df.columns
    cols.remove(old_name)
    return df.select(*cols, f.col(old_name).alias(new_name))


def flagged_purchases(df):
    return df.select(*df.columns, f.when(f.col("Invoice").startswith("C"), f.lit("C")).otherwise(f.lit("P"))
                     .alias("Cancel_flag"))


def add_time_columns(df):
    return df.select(*df.columns, f.date_format("InvoiceDate", "yyyy-MM").alias("Month"),
                     f.date_format("InvoiceDate", "yyyy-MM-dd").alias("Day"))


def add_postage_info(df):
    return df.select(*df.columns, f.when(f.col("StockCode") == "POST",
                                         f.lit("Postage")).otherwise("No_postage").alias("Postage"))


def compute_spend(df):
    return df.select(*df.columns, (f.col("Quantity") * f.col("Price")).alias("spend"))


def flag_uk(df):
    return df.select(*df.columns, f.when(f.col("Country") == 'United Kingdom', 1).otherwise(0).alias("UK"))


def keep_rows_with_customer_id(df):
    return df.filter("CustomerId is not null")


def get_users_info(df):
    return df.groupby("CustomerId", "month") \
        .agg(f.min("day").alias("FirstPurchaseDate"), f.max("day").alias("LastPurchaseDate"))


def get_customer_list(df):
    return df.select("CustomerId").distinct()


def get_purchasers(df, month=None):
    return df.filter(f"month='{month}'").select("CustomerId").distinct()


def get_non_buyer(df, month=None):
    buyers = get_purchasers(df, month)
    all_customers = get_customer_list(df)
    return all_customers.join(f.broadcast(buyers), all_customers.CustomerId == buyers.CustomerId, "leftanti")


def get_purchaser_data(df, month):
    return df.filter(f"month<'{month}'").join(get_purchasers(df, month=month), "CustomerId")


def get_non_buyer_data(df, month):
    buyers = get_purchasers(df, month)
    return df.filter(f"month<'{month}'").join(f.broadcast(buyers), df.CustomerId == buyers.CustomerId, "leftanti")


@f.udf(t.FloatType())
def get_buying_frequency(all_buying_date):
    from dateutil import parser
    sum_diff = 0
    all_buying_date.sort()
    for i in range(len(all_buying_date) - 1):
        # print(f"Before summing: begin {all_buying_date[i+1]} end {all_buying_date[i]} sum_diff {sum_diff}")
        sum_diff += (parser.parse(all_buying_date[i+1]) - parser.parse(all_buying_date[i])).days
        # print(f"After summing: begin {all_buying_date[i + 1]} end {all_buying_date[i]} sum_diff {sum_diff}")
    # print(f"Nombre de jours d'achat: {len(all_buying_date)}, total de différence de jous entre 2 achats: {sum_diff} et moyenne : {sum_diff/len(all_buying_date)}")
    return sum_diff/len(all_buying_date)


def prepare_data(df, old_name="Customer Id", new_name="CustomerId"):
    df = rename_column(df, old_name, new_name)
    df = keep_rows_with_customer_id(df)
    df = flagged_purchases(df)
    df = add_time_columns(df)
    df = add_postage_info(df)
    df = compute_spend(df)
    df = flag_uk(df)
    # df = get_bill_period(df, "2009-12-01", "2011-11-01")
    df_info = get_users_info(df)
    print(df.columns)
    return df


