"""
Centralized repository for Spark SQL queries used in the application.
Refactored to use PySpark DataFrame API.
"""
from pyspark.sql.functions import count, countDistinct, avg, date_format, col, desc

def get_user_metrics_df(df):
    """
    Returns DataFrame with total users and total posts.
    """
    return df.agg(
        countDistinct("Subject ID").alias("num_users"),
        count("*").alias("total_posts")
    )

def get_depression_averages_df(df):
    """
    Returns DataFrame with average depression risks.
    """
    return df.agg(
        avg("Prob_Severe_Depressed").alias("avg_severe"),
        avg("Prob_Moderate_Depressed").alias("avg_moderate")
    )

def get_posts_over_time_df(df):
    """
    Returns DataFrame with posts count grouped by month.
    """
    return df.withColumn("MonthDate", date_format("Date", 'yyyy-MM-01')) \
             .groupBy("MonthDate") \
             .agg(count("*").alias("Posts")) \
             .orderBy("MonthDate")

def get_top_active_users_df(df, limit=10):
    """
    Returns DataFrame with top users by post count.
    """
    return df.groupBy(col("Subject ID").alias("User_ID")) \
             .agg(count("*").alias("Post_Count")) \
             .orderBy(desc("Post_Count")) \
             .limit(limit)
