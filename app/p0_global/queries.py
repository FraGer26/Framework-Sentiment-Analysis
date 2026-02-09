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

def get_weekday_activity_df(df):
    """
    Distribuzione post per giorno della settimana.
    Restituisce: DayOfWeek (1=Domenica, 7=Sabato), Posts
    """
    from pyspark.sql.functions import dayofweek
    return df.withColumn("DayOfWeek", dayofweek("Date")) \
        .groupBy("DayOfWeek") \
        .agg(count("*").alias("Posts")) \
        .orderBy("DayOfWeek")

def get_post_length_stats_df(df):
    """
    Statistiche sulla lunghezza dei post (caratteri).
    Restituisce: Avg_Length, Min_Length, Max_Length
    """
    from pyspark.sql.functions import length, min as spark_min, max as spark_max
    return df.withColumn("text_len", length("Text")) \
        .agg(avg("text_len").alias("Avg_Length"),
             spark_min("text_len").alias("Min_Length"),
             spark_max("text_len").alias("Max_Length"))

def get_activity_risk_correlation_df(df, limit=20):
    """
    Utenti ordinati per rischio e attività, mostra correlazione.
    Restituisce: Subject_ID, Post_Count, Avg_Prob_Severe_Depressed
    """
    return df.groupBy(col("Subject ID").alias("Subject_ID")) \
        .agg(count("*").alias("Post_Count"),
             avg("Prob_Severe_Depressed").alias("Avg_Prob_Severe_Depressed")) \
        .orderBy(desc("Avg_Prob_Severe_Depressed"), desc("Post_Count")) \
        .limit(limit)
