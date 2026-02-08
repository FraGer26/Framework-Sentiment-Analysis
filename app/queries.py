"""
Repository centralizzato per query Spark SQL usate nell'applicazione.
Rifattorizzato per usare l'API PySpark DataFrame.
"""
from pyspark.sql.functions import count, countDistinct, avg, date_format, col, desc
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def get_user_metrics_df(df):
    """
    Restituisce DataFrame con totale utenti e totale post.
    """
    return df.agg(
        countDistinct("Subject ID").alias("num_users"),
        count("*").alias("total_posts")
    )

def get_depression_averages_df(df):
    """
    Restituisce DataFrame con rischi medi di depressione.
    """
    return df.agg(
        avg("Prob_Severe_Depressed").alias("avg_severe"),
        avg("Prob_Moderate_Depressed").alias("avg_moderate")
    )

def get_posts_over_time_df(df):
    """
    Restituisce DataFrame con conteggio post raggruppati per mese.
    """
    return df.withColumn("MonthDate", date_format("Date", 'yyyy-MM-01')) \
             .groupBy("MonthDate") \
             .agg(count("*").alias("Posts")) \
             .orderBy("MonthDate")

def get_top_active_users_df(df, limit=10):
    """
    Restituisce DataFrame con utenti top per numero di post.
    """
    return df.groupBy(col("Subject ID").alias("User_ID")) \
             .agg(count("*").alias("Post_Count")) \
             .orderBy(desc("Post_Count")) \
             .limit(limit)
