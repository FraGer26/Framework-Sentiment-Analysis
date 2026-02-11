# =============================================================================
# MODULO: queries.py
# DESCRIZIONE: Repository centralizzato per tutte le query Spark SQL
#              utilizzate nell'applicazione per l'analisi del dataset.
#              Ogni funzione riceve un DataFrame Spark e restituisce
#              un nuovo DataFrame Spark con i risultati della query.
# =============================================================================

# Importa le funzioni Spark SQL necessarie per le aggregazioni e trasformazioni
from pyspark.sql.functions import count, countDistinct, avg, date_format, col, desc, dayofweek, length
# Importa min e max con alias per evitare conflitto con le funzioni built-in di Python
from pyspark.sql.functions import min as spark_min, max as spark_max
# Importa i tipi di dato per la definizione degli schemi Spark
from pyspark.sql.types import StructType, StructField, StringType, FloatType


def get_user_metrics_df(df):
    """
    Calcola le metriche principali sugli utenti del dataset.
    
    Args:
        df: DataFrame Spark contenente i dati del dataset
        
    Returns:
        DataFrame Spark con colonne: num_users, total_posts
    """
    # Esegue un'aggregazione globale sul DataFrame:
    # - countDistinct("Subject ID"): conta il numero di utenti unici nel dataset
    # - count("*"): conta il numero totale di righe (post) nel dataset
    return df.agg(
        countDistinct("Subject ID").alias("num_users"),  # Numero utenti unici
        count("*").alias("total_posts")  # Numero totale di post
    )


def get_depression_averages_df(df):
    """
    Calcola le medie delle probabilità di depressione sull'intero dataset.
    
    Args:
        df: DataFrame Spark con colonne Prob_Severe_Depressed e Prob_Moderate_Depressed
        
    Returns:
        DataFrame Spark con colonne: avg_severe, avg_moderate
    """
    # Calcola la media aritmetica delle due colonne di probabilità di depressione:
    # - avg("Prob_Severe_Depressed"): media della probabilità di depressione severa
    # - avg("Prob_Moderate_Depressed"): media della probabilità di depressione moderata
    return df.agg(
        avg("Prob_Severe_Depressed").alias("avg_severe"),  # Media rischio severo
        avg("Prob_Moderate_Depressed").alias("avg_moderate")  # Media rischio moderato
    )


def get_posts_over_time_df(df):
    """
    Raggruppa i post per mese e conta quanti ce ne sono per ogni mese.
    
    Args:
        df: DataFrame Spark con colonna Date
        
    Returns:
        DataFrame Spark con colonne: MonthDate, Posts (ordinato cronologicamente)
    """
    # Aggiunge una colonna "MonthDate" formattando la data come primo giorno del mese (es. "2024-01-01")
    # Poi raggruppa per mese, conta i post e ordina cronologicamente
    return df.withColumn("MonthDate", date_format("Date", 'yyyy-MM-01')) \
             .groupBy("MonthDate") \
             .agg(count("*").alias("Posts")) \
             .orderBy("MonthDate")


def get_top_active_users_df(df, limit=10):
    """
    Trova gli utenti più attivi in base al numero di post pubblicati.
    
    Args:
        df: DataFrame Spark con colonna Subject ID
        limit (int): Numero massimo di utenti da restituire (default: 10)
        
    Returns:
        DataFrame Spark con colonne: User_ID, Post_Count (ordinato per attività decrescente)
    """
    # Raggruppa per ID utente, conta i post di ciascuno,
    # ordina per numero di post decrescente e prende i primi N utenti
    return df.groupBy(col("Subject ID").alias("User_ID")) \
             .agg(count("*").alias("Post_Count")) \
             .orderBy(desc("Post_Count")) \
             .limit(limit)


def get_weekday_activity_df(df):
    """
    Analizza la distribuzione dei post per giorno della settimana.
    Utile per capire in quali giorni gli utenti sono più attivi.
    
    Args:
        df: DataFrame Spark con colonna Date
        
    Returns:
        DataFrame Spark con colonne: DayOfWeek (1=Domenica, 7=Sabato), Posts
    """
    # Aggiunge una colonna con il numero del giorno della settimana (1=Dom, 7=Sab)
    # Raggruppa per giorno, conta i post e ordina per giorno
    return df.withColumn("DayOfWeek", dayofweek("Date")) \
        .groupBy("DayOfWeek") \
        .agg(count("*").alias("Posts")) \
        .orderBy("DayOfWeek")


def get_post_length_stats_df(df):
    """
    Calcola statistiche sulla lunghezza dei post (in numero di caratteri).
    
    Args:
        df: DataFrame Spark con colonna Text
        
    Returns:
        DataFrame Spark con colonne: Avg_Length, Min_Length, Max_Length
    """
    # Aggiunge una colonna con la lunghezza in caratteri di ogni post
    # Poi calcola media, minimo e massimo della lunghezza
    return df.withColumn("text_len", length("Text")) \
        .agg(avg("text_len").alias("Avg_Length"),  # Lunghezza media dei post
             spark_min("text_len").alias("Min_Length"),  # Post più corto
             spark_max("text_len").alias("Max_Length"))  # Post più lungo


def get_activity_risk_correlation_df(df, limit=20):
    """
    Analizza la correlazione tra attività (numero di post) e rischio di depressione.
    Mostra i top N utenti ordinati per rischio decrescente.
    
    Args:
        df: DataFrame Spark con colonne Subject ID e Prob_Severe_Depressed
        limit (int): Numero massimo di utenti da restituire (default: 20)
        
    Returns:
        DataFrame Spark con colonne: Subject_ID, Post_Count, Avg_Prob_Severe_Depressed
    """
    # Per ogni utente calcola il numero di post e la media della probabilità di depressione severa
    # Ordina per rischio decrescente (e a parità di rischio, per numero di post decrescente)
    return df.groupBy(col("Subject ID").alias("Subject_ID")) \
        .agg(count("*").alias("Post_Count"),  # Conta i post dell'utente
             avg("Prob_Severe_Depressed").alias("Avg_Prob_Severe_Depressed")) \
        .orderBy(desc("Avg_Prob_Severe_Depressed"), desc("Post_Count")) \
        .limit(limit)  # Limita ai primi N utenti
