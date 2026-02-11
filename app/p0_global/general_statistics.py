# =============================================================================
# MODULO: general_statistics.py
# DESCRIZIONE: Calcola statistiche globali sul dataset utilizzando Apache Spark
#              per elaborazioni distribuite ad alte prestazioni.
# =============================================================================

import pandas as pd  # Libreria per manipolazione dati tabulari
import os  # Libreria per operazioni su file system e percorsi
from p0_global import data, queries  # Importa moduli locali: data per costanti, queries per query Spark
from p1_segmentation import ema  # Importa modulo EMA per calcolo media mobile esponenziale

# =============================================================================
# SEZIONE: IMPORTAZIONI SPARK
# Importazioni necessarie per l'integrazione con Apache Spark
# =============================================================================
from pyspark.sql import SparkSession  # Classe principale per creare sessioni Spark
from pyspark.sql.functions import col, date_format, to_date, count, avg, desc  # Funzioni Spark SQL
from pyspark.sql.types import StructType, StructField, StringType, FloatType  # Tipi di dato Spark


def get_spark_session():
    """
    Crea e restituisce una sessione Spark configurata per l'applicazione.
    
    Questa funzione:
    1. Configura le opzioni JVM per compatibilità con Java 11+
    2. Imposta le variabili d'ambiente necessarie per PySpark
    3. Crea una SparkSession con configurazioni ottimizzate
    
    Returns:
        SparkSession: Sessione Spark configurata e pronta all'uso
    """
    # Imposta opzioni JVM per permettere l'accesso a moduli Java interni
    # Necessario per versioni Java 17+ che hanno restrizioni di accesso
    os.environ['SPARK_SUBMIT_OPTS'] = '--add-opens java.base/javax.security.auth=ALL-UNNAMED'
    
    # Configura argomenti per l'esecuzione di PySpark con opzioni JVM aggiuntive
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="--add-opens java.base/javax.security.auth=ALL-UNNAMED" pyspark-shell'
    
    # Costruisce e configura la SparkSession usando il pattern Builder
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
    Calcola statistiche globali sul dataset utilizzando Spark SQL.
    
    Questa funzione esegue 7 query Spark per analizzare il dataset:
    1. Metriche utente (conteggio utenti e post)
    2. Medie probabilità depressione (severa e moderata)
    3. Distribuzione post nel tempo (per mese)
    4. Classifica utenti più attivi (top 10)
    5. Distribuzione attività per giorno della settimana
    6. Statistiche lunghezza dei post
    7. Correlazione tra attività e rischio depressione
    
    I risultati vengono salvati in cache come file JSON per evitare
    ricalcoli costosi nelle sessioni successive.
    
    Args:
        df (pd.DataFrame): DataFrame Pandas con i dati del dataset
        
    Returns:
        tuple: 7 DataFrame Pandas con i risultati delle query
    """
    
    # ==========================================================================
    # DEFINIZIONE PERCORSI CACHE
    # I file JSON vengono salvati nella directory di cache globale
    # ==========================================================================
    cache_metrics_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_metrics.json")  # Metriche utenti
    cache_avgs_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_avgs.json")  # Medie depressione
    cache_time_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_time.json")  # Serie temporale
    cache_top_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_top_users.json")  # Top utenti
    
    # ==========================================================================
    # INIZIALIZZAZIONE SPARK E CONVERSIONE DATI
    # ==========================================================================
    
    # Ottiene o crea una sessione Spark singleton
    spark = get_spark_session()
    
    # Crea una copia del DataFrame per non modificare l'originale
    df_clean = df.copy()
    
    # Converte la colonna Date in stringa per compatibilità Spark
    # Spark richiede tipi serializzabili per la conversione da Pandas
    if 'Date' in df_clean.columns:
        df_clean['Date'] = df_clean['Date'].astype(str)
    
    # Converte il DataFrame Pandas in DataFrame Spark
    # Questo trasferisce i dati nel cluster Spark per elaborazione distribuita
    sdf = spark.createDataFrame(df_clean)
    
    # Riconverte la colonna Date da stringa a tipo Date di Spark
    # Necessario per operazioni temporali come groupBy per mese
    sdf = sdf.withColumn("Date", to_date(col("Date")))
    
    # ==========================================================================
    # QUERY 1: METRICHE UTENTE
    # Conta il numero totale di utenti unici e il numero totale di post
    # ==========================================================================
    metrics_df = queries.get_user_metrics_df(sdf).toPandas()  # Esegue query e converte in Pandas
    metrics_df.to_json(cache_metrics_path)  # Salva risultato in cache JSON
    
    # ==========================================================================
    # QUERY 2: MEDIE PROBABILITÀ DEPRESSIONE
    # Calcola la media delle probabilità di depressione severa e moderata
    # ==========================================================================
    avgs_df = None  # Inizializza a None nel caso la colonna non esista
    if "Prob_Severe_Depressed" in df.columns:  # Verifica che la colonna esista
        avgs_df = queries.get_depression_averages_df(sdf).toPandas()  # Esegue query Spark
        avgs_df.to_json(cache_avgs_path)  # Salva in cache
    
    # ==========================================================================
    # QUERY 3: DISTRIBUZIONE POST NEL TEMPO
    # Raggruppa i post per mese e conta quanti ce ne sono per ogni mese
    # ==========================================================================
    time_df = None  # Inizializza a None
    if 'Date' in df.columns:  # Verifica che la colonna Date esista
        time_df = queries.get_posts_over_time_df(sdf).toPandas()  # Esegue query temporale
        time_df.to_json(cache_time_path)  # Salva in cache
        
    # ==========================================================================
    # QUERY 4: UTENTI PIÙ ATTIVI
    # Trova i 10 utenti con il maggior numero di post
    # ==========================================================================
    top_activity = queries.get_top_active_users_df(sdf).toPandas()  # Esegue query classifica
    top_activity.to_json(cache_top_path)  # Salva in cache
    
    # ==========================================================================
    # NUOVE ANALYTICS CON CACHE
    # Definizione dei percorsi per le nuove metriche
    # ==========================================================================
    cache_weekday_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_weekday.json")  # Per giorno settimana
    cache_length_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_length.json")  # Per lunghezza post
    cache_correlation_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_correlation.json")  # Per correlazione
    
    # ==========================================================================
    # QUERY 5: ATTIVITÀ PER GIORNO DELLA SETTIMANA
    # Conta i post per ogni giorno della settimana (1=Dom, 7=Sab)
    # ==========================================================================
    weekday_df = None  # Inizializza a None
    if 'Date' in df.columns:  # Verifica presenza colonna Date
        weekday_df = queries.get_weekday_activity_df(sdf).toPandas()  # Esegue query per giorno
        # Mappa i numeri dei giorni ai loro nomi abbreviati in italiano
        day_names = {1: 'Dom', 2: 'Lun', 3: 'Mar', 4: 'Mer', 5: 'Gio', 6: 'Ven', 7: 'Sab'}
        weekday_df['DayName'] = weekday_df['DayOfWeek'].map(day_names)  # Aggiunge colonna nomi
        weekday_df.to_json(cache_weekday_path)  # Salva in cache
    
    # ==========================================================================
    # QUERY 6: STATISTICHE LUNGHEZZA POST
    # Calcola media, minimo e massimo della lunghezza dei post in caratteri
    # ==========================================================================
    length_df = queries.get_post_length_stats_df(sdf).toPandas()  # Esegue query lunghezza
    length_df.to_json(cache_length_path)  # Salva in cache
    
    # ==========================================================================
    # QUERY 7: CORRELAZIONE ATTIVITÀ-RISCHIO
    # Per ogni utente, calcola numero post e media probabilità depressione
    # Ordina per rischio decrescente, prende i top 20
    # ==========================================================================
    correlation_df = None  # Inizializza a None
    if 'Prob_Severe_Depressed' in df.columns:  # Verifica che la colonna esista
        correlation_df = queries.get_activity_risk_correlation_df(sdf).toPandas()  # Esegue query
        correlation_df.to_json(cache_correlation_path)  # Salva in cache
    
    # Restituisce tutti i 7 DataFrame come tupla
    return metrics_df, avgs_df, time_df, top_activity, weekday_df, length_df, correlation_df


def compute_risk_rankings(df, half_life=15):
    """
    Calcola la classifica degli utenti più a rischio usando EMA (Media Mobile Esponenziale).
    
    Questa funzione NON usa Spark perché l'EMA richiede elaborazione sequenziale
    per ogni utente (ogni valore dipende dal precedente). Viene quindi usato
    Pandas puro per iterare su ogni utente e calcolare la sua serie EMA.
    
    L'EMA (Exponential Moving Average) con half-life dà più peso ai valori recenti,
    permettendo di identificare utenti il cui rischio sta aumentando.
    
    Args:
        df (pd.DataFrame): DataFrame con i dati di tutti gli utenti
        half_life (int): Periodo di dimezzamento per l'EMA in giorni (default: 15)
        
    Returns:
        pd.DataFrame: Top 20 utenti più a rischio con colonne:
                      - User ID: Identificativo utente
                      - Current Risk: Ultimo valore EMA del rischio
                      - Avg Risk (EMA): Media della serie EMA
                      - Peak Risk: Valore massimo EMA raggiunto
    """
    
    # Definisce il nome del file cache basato sul parametro half_life
    cache_filename = f"global_rank_h{half_life}.json"
    cache_path = os.path.join(data.GLOBAL_CACHE_DIR, cache_filename)  # Percorso completo cache
    
    # Gestisce il caso di DataFrame vuoto
    if df.empty:
        # Restituisce un DataFrame vuoto con le colonne corrette
        return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])

    user_stats = []  # Lista per accumulare le statistiche di ogni utente
    
    # ==========================================================================
    # RAGGRUPPAMENTO PER UTENTE
    # Divide il DataFrame in gruppi, uno per ogni utente unico
    # ==========================================================================
    grouped = df.groupby("Subject ID")  # Raggruppa per ID utente
    
    # ==========================================================================
    # ITERAZIONE SU OGNI UTENTE
    # Per ogni utente, calcola la serie EMA e estrae le statistiche
    # ==========================================================================
    for user_id, user_df in grouped:  # Itera su ogni gruppo (utente)
        # Calcola la serie temporale EMA per questo utente
        # smoothing=False per non applicare ulteriore smoothing alla serie
        r_series = ema.compute_risk_series(user_df, half_life=half_life, smoothing=False)
        
        # Verifica che la serie non sia vuota
        if not r_series.empty:
            # Estrae l'ultimo valore della serie (rischio corrente)
            curr_risk = float(r_series.iloc[-1])
            # Calcola la media di tutti i valori EMA
            avg_risk = float(r_series.mean())
            # Trova il valore massimo EMA (picco di rischio)
            max_risk = float(r_series.max())
            
            # Aggiunge le statistiche dell'utente alla lista
            user_stats.append({
                "User ID": str(user_id),  # Converte ID a stringa per uniformità
                "Current Risk": curr_risk,  # Rischio attuale
                "Avg Risk (EMA)": avg_risk,  # Media del rischio
                "Peak Risk": max_risk  # Picco massimo di rischio
            })
    
    # ==========================================================================
    # CREAZIONE RISULTATO FINALE
    # ==========================================================================
    
    # Gestisce il caso in cui nessun utente abbia dati validi
    if not user_stats:
        return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])
    
    # Converte la lista di dizionari in DataFrame Pandas
    results_df = pd.DataFrame(user_stats)
    
    # Ordina per rischio corrente decrescente e prende i primi 20
    top_risky = results_df.sort_values("Current Risk", ascending=False).head(20)
    
    # ==========================================================================
    # SALVATAGGIO IN CACHE
    # ==========================================================================
    try:
        top_risky.to_json(cache_path)  # Salva in formato JSON
    except Exception:
        pass  # Ignora errori di salvataggio (es. permessi file)
    
    # Restituisce il DataFrame con i top 20 utenti a rischio
    return top_risky
