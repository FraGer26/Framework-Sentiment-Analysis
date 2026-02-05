"""
Centralized repository for Spark SQL queries used in the application.
"""

def get_user_metrics_query(table_name="reddit_posts"):
    """
    Returns SQL query to calculate total users and total posts.
    """
    return f"""
        SELECT 
            count(DISTINCT `Subject ID`) as num_users,
            count(*) as total_posts
        FROM {table_name}
    """

def get_depression_averages_query(table_name="reddit_posts"):
    """
    Returns SQL query to calculate average depression risks.
    """
    return f"""
        SELECT 
            avg(Prob_Severe_Depressed) as avg_severe,
            avg(Prob_Moderate_Depressed) as avg_moderate
        FROM {table_name}
    """

def get_posts_over_time_query(table_name="reddit_posts"):
    """
    Returns SQL query to calculate posts count grouped by month.
    """
    return f"""
        SELECT 
            date_format(Date, 'yyyy-MM-01') as MonthDate,
            count(*) as Posts
        FROM {table_name}
        GROUP BY 1
        ORDER BY 1
    """

def get_top_active_users_query(table_name="reddit_posts", limit=10):
    """
    Returns SQL query to find top users by post count.
    """
    return f"""
        SELECT `Subject ID` as User_ID, count(*) as Post_Count
        FROM {table_name}
        GROUP BY `Subject ID`
        ORDER BY Post_Count DESC
        LIMIT {limit}
    """
