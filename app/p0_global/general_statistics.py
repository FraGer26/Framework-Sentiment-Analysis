import pandas as pd
import os
from p0_global import data, queries
from p1_segmentation import ema

# --- Integrazione Spark ---
# Inizializzazione Spark lazy
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, to_date, count, avg, desc
from pyspark.sql.types import StructType, StructField, StringType, FloatType

def get_spark_session():
    import os
    # Imposta opzioni JVM per compatibilità Java 17+
    os.environ['SPARK_SUBMIT_OPTS'] = '--add-opens java.base/javax.security.auth=ALL-UNNAMED'
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="--add-opens java.base/javax.security.auth=ALL-UNNAMED" pyspark-shell'
    
    return SparkSession.builder \
        .appName("RedditAnalyticsApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.extraJavaOptions", "--add-opens java.base/javax.security.auth=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.lang.invoke=ALL-UNNAMED --add-opens java.base/java.io=ALL-UNNAMED") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.heartbeatInterval", "100s") \
        .config("spark.network.timeout", "300s") \
        .getOrCreate()

def compute_global_stats(df):
    """
    Calcola statistiche globali usando Spark e restituisce DataFrame Pandas.
    Gestisce anche la cache JSON.
    """
    # --- Percorsi Cache ---
    cache_metrics_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_metrics.json")
    cache_avgs_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_avgs.json")
    cache_time_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_time.json")
    cache_top_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_top_users.json")
    
    # 0. Conversione Payload a Spark
    spark = get_spark_session()
    
    # Pulizia nomi colonne per compatibilità Spark
    df_clean = df.copy()
    if 'Date' in df_clean.columns:
        df_clean['Date'] = df_clean['Date'].astype(str)
    
    # Creazione DataFrame Spark
    sdf = spark.createDataFrame(df_clean)
    sdf = sdf.withColumn("Date", to_date(col("Date")))
    
    # 1. Metriche Utente
    metrics_df = queries.get_user_metrics_df(sdf).toPandas()
    metrics_df.to_json(cache_metrics_path)
    
    # 2. Medie Depressione
    avgs_df = None
    if "Prob_Severe_Depressed" in df.columns:
        avgs_df = queries.get_depression_averages_df(sdf).toPandas()
        avgs_df.to_json(cache_avgs_path)
    
    # 3. Post nel Tempo
    time_df = None
    if 'Date' in df.columns:
        time_df = queries.get_posts_over_time_df(sdf).toPandas()
        time_df.to_json(cache_time_path)
        
    # 4. Utenti più Attivi
    top_activity = queries.get_top_active_users_df(sdf).toPandas()
    top_activity.to_json(cache_top_path)
    
    return metrics_df, avgs_df, time_df, top_activity

def compute_risk_rankings(df, half_life=15):
    """
    Calcola gli utenti più a rischio usando Pure Pandas.
    Itera sugli utenti, calcola l'EMA (tramite ema.py) e seleziona i top 20.
    """
    cache_filename = f"global_rank_h{half_life}.json" 
    cache_path = os.path.join(data.GLOBAL_CACHE_DIR, cache_filename)
    
    if df.empty:
        return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])

    user_stats = []
    
    # Raggruppa per Subject ID
    grouped = df.groupby("Subject ID")
    
    for user_id, user_df in grouped:
        # Calcola serie EMA (senza smoothing come richiesto)
        r_series = ema.compute_risk_series(user_df, half_life=half_life, smoothing=False)
        
        if not r_series.empty:
            curr_risk = float(r_series.iloc[-1])
            avg_risk = float(r_series.mean())
            max_risk = float(r_series.max())
            
            user_stats.append({
                "User ID": str(user_id),
                "Current Risk": curr_risk,
                "Avg Risk (EMA)": avg_risk,
                "Peak Risk": max_risk
            })
            
    if not user_stats:
        return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])
        
    results_df = pd.DataFrame(user_stats)
    
    # Ordina per Current Risk decrescente e prendi i top 20
    top_risky = results_df.sort_values("Current Risk", ascending=False).head(20)
    
    # Cache
    try:
        top_risky.to_json(cache_path)
    except Exception:
        pass
        
    return top_risky
