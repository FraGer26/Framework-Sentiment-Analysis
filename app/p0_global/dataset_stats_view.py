# =============================================================================
# MODULO: dataset_stats_view.py
# DESCRIZIONE: Renderizza la pagina delle statistiche globali del dataset.
#              Mostra metriche, grafici e analytics calcolate con Spark SQL,
#              organizzate in 4 tab: Overview, Rankings, New Analytics, GPT.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import pandas as pd  # Libreria per manipolazione dati tabulari
import plotly.graph_objects as go  # Libreria per grafici interattivi
import os  # Libreria per operazioni su file e percorsi
from p0_global import data, general_statistics  # Moduli locali per dati e statistiche Spark
from p3_llm_judge import gpt_evaluator  # Modulo per la valutazione GPT dei report


def render_dataset_statistics(df, api_key):
    """
    Renderizza la pagina completa delle statistiche globali del dataset.
    
    Gestisce la cache dei risultati Spark e organizza la visualizzazione in 4 tab:
    1. General Overview: metriche utente, distribuzione depressione, volume post
    2. Rankings & Risk: utenti più attivi e più a rischio (EMA)
    3. New Analytics: attività settimanale, lunghezza post, correlazione
    4. GPT Evaluation: valutazione qualitativa dei report con LLM
    
    Args:
        df (pd.DataFrame): DataFrame completo del dataset
        api_key (str): Chiave API OpenAI per le valutazioni GPT
    """
    # Titolo principale della sezione
    st.subheader("🌍 Global Dataset Statistics (Powered by Spark SQL)")
    
    # ==========================================================================
    # DEFINIZIONE PERCORSI CACHE
    # Ogni statistica è salvata come file JSON nella directory di cache globale
    # ==========================================================================
    cache_metrics_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_metrics.json")  # Metriche utenti
    cache_avgs_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_avgs.json")  # Medie depressione
    cache_time_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_time.json")  # Serie temporale
    cache_top_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_top_users.json")  # Top utenti attivi
    cache_weekday_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_weekday.json")  # Attività settimanale
    cache_length_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_length.json")  # Lunghezza post
    cache_correlation_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_correlation.json")  # Correlazione
    
    # ==========================================================================
    # VERIFICA ESISTENZA CACHE
    # Controlla se tutti i file cache base esistono
    # ==========================================================================
    caches_exist = (
        os.path.exists(cache_metrics_path) and  # Esiste cache metriche?
        os.path.exists(cache_avgs_path) and  # Esiste cache medie?
        os.path.exists(cache_time_path) and  # Esiste cache temporale?
        os.path.exists(cache_top_path)  # Esiste cache top utenti?
    )
    
    # Pulsante per forzare il ricalcolo delle statistiche
    recalc = st.button("🔄 Recalculate Global Statistics")
    
    # ==========================================================================
    # INIZIALIZZAZIONE VARIABILI
    # Tutti i DataFrame partono a None e vengono popolati dalla cache o dal calcolo
    # ==========================================================================
    metrics_df = None  # Metriche utente (num_users, total_posts)
    avgs_df = None  # Medie depressione (avg_severe, avg_moderate)
    time_df = None  # Post nel tempo (MonthDate, Posts)
    top_activity = None  # Utenti più attivi (User_ID, Post_Count)
    weekday_df = None  # Attività per giorno settimana (DayOfWeek, Posts)
    length_df = None  # Statistiche lunghezza (Avg_Length, Min_Length, Max_Length)
    correlation_df = None  # Correlazione attività-rischio (Subject_ID, Post_Count, Avg_Prob)
    
    # ==========================================================================
    # CARICAMENTO DALLA CACHE
    # Se la cache esiste e l'utente non ha premuto Recalculate, carica da file
    # ==========================================================================
    if caches_exist and not recalc:
        try:
            # Carica i 4 DataFrame base dalla cache JSON
            metrics_df = pd.read_json(cache_metrics_path)  # Carica metriche
            avgs_df = pd.read_json(cache_avgs_path)  # Carica medie
            time_df = pd.read_json(cache_time_path).sort_values("MonthDate")  # Carica e ordina temporale
            top_activity = pd.read_json(cache_top_path)  # Carica top utenti
            
            # Carica le nuove analytics dalla cache (se esistono)
            if os.path.exists(cache_weekday_path):
                weekday_df = pd.read_json(cache_weekday_path)  # Carica attività settimanale
            if os.path.exists(cache_length_path):
                length_df = pd.read_json(cache_length_path)  # Carica statistiche lunghezza
            if os.path.exists(cache_correlation_path):
                correlation_df = pd.read_json(cache_correlation_path)  # Carica correlazione
            
            # Mostra messaggio di successo
            st.success("Loaded statistics from cache.")
        except Exception as e:
            # Se il caricamento fallisce, segnala e forza il ricalcolo
            st.warning(f"Cache load failed: {e}. Recalculating...")
            caches_exist = False  # Forza il ricalcolo

    # ==========================================================================
    # CALCOLO CON SPARK
    # Se la cache non esiste o l'utente ha premuto Recalculate, esegue le query
    # ==========================================================================
    if not caches_exist or recalc:
        # Mostra spinner di caricamento durante il calcolo
        with st.spinner("Initializing Spark and calculating statistics... (This may take a moment)"):
            # Chiama la funzione che esegue tutte le 7 query Spark
            metrics_df, avgs_df, time_df, top_activity, weekday_df, length_df, correlation_df = general_statistics.compute_global_stats(df)
            st.success("Statistics calculated and cached.")  # Conferma completamento
    
    # ==========================================================================
    # ORGANIZZAZIONE IN TAB
    # I risultati vengono mostrati in 4 tab tematici
    # ==========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 General Overview",  # Tab 1: metriche generali
        "🏆 Rankings & Risk",  # Tab 2: classifiche utenti
        "📅 New Analytics",  # Tab 3: analisi aggiuntive
        "⚖️ GPT Evaluation"  # Tab 4: valutazione LLM
    ])
    
    # ==========================================================================
    # TAB 1: GENERAL OVERVIEW
    # Mostra metriche utente, distribuzione depressione e volume post nel tempo
    # ==========================================================================
    with tab1:
        # --- Sezione Metriche Utente ---
        if metrics_df is not None and not metrics_df.empty:
            # Estrae i valori dal DataFrame delle metriche
            num_users = metrics_df['num_users'][0]  # Numero utenti unici
            total_posts = metrics_df['total_posts'][0]  # Numero totale post
            # Calcola la media di post per utente (evita divisione per zero)
            avg_posts_user = total_posts / num_users if num_users > 0 else 0
            
            # Mostra le metriche in 3 colonne affiancate
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", f"{num_users}")  # Metrica: utenti totali
            col2.metric("Total Posts", f"{total_posts}")  # Metrica: post totali
            col3.metric("Avg Posts/User", f"{avg_posts_user:.1f}")  # Metrica: media post/utente
        
        st.markdown("---")  # Separatore orizzontale
        
        # --- Sezione Distribuzione Probabilità Depressione ---
        st.markdown("### Depression Probability (Dataset-wide)")
        
        # Verifica che le colonne di probabilità esistano nel dataset
        has_dep_cols = "Prob_Severe_Depressed" in df.columns and "Prob_Moderate_Depressed" in df.columns
        
        if has_dep_cols:
            # Mostra le medie delle probabilità di depressione
            if avgs_df is not None and not avgs_df.empty:
                avg_severe = avgs_df['avg_severe'][0]  # Media depressione severa
                avg_moderate = avgs_df['avg_moderate'][0]  # Media depressione moderata
                avg_none = 1.0 - (avg_severe + avg_moderate)  # Stima probabilità non-depresso
                
                # Mostra le 3 medie in colonne affiancate
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Severe Score", f"{avg_severe:.4f}")  # Media severa (4 decimali)
                c2.metric("Avg Moderate Score", f"{avg_moderate:.4f}")  # Media moderata
                c3.metric("Avg Non-Depressed (Est.)", f"{avg_none:.4f}")  # Stima non-depresso
            
            # --- Istogramma delle distribuzioni di rischio ---
            fig_hist = go.Figure()  # Crea figura vuota
            # Aggiunge istogramma per la probabilità severa
            fig_hist.add_trace(go.Histogram(
                x=df["Prob_Severe_Depressed"],  # Dati: probabilità severa
                name='Severe',  # Nome nella legenda
                opacity=0.75  # Trasparenza al 75% per sovrapporre i due istogrammi
            ))
            # Aggiunge istogramma per la probabilità moderata
            fig_hist.add_trace(go.Histogram(
                x=df["Prob_Moderate_Depressed"],  # Dati: probabilità moderata
                name='Moderate',  # Nome nella legenda
                opacity=0.75  # Stessa trasparenza
            ))
            # Configura il layout con sovrapposizione delle barre
            fig_hist.update_layout(
                barmode='overlay',  # Sovrappone i due istogrammi
                title="Distribution of Risk Scores",  # Titolo del grafico
                xaxis_title="Score",  # Etichetta asse X
                yaxis_title="Count"  # Etichetta asse Y
            )
            st.plotly_chart(fig_hist, use_container_width=True)  # Mostra il grafico
        else:
            # Avviso se le colonne necessarie non esistono
            st.warning("Depression probability columns not found.")
            
        st.markdown("---")  # Separatore
        
        # --- Sezione Volume Post nel Tempo ---
        st.markdown("### 📈 Posts Volume Over Time")
        if 'Date' in df.columns and time_df is not None:
            # Crea grafico a barre con i post raggruppati per mese
            fig_time = go.Figure(data=[
                go.Bar(
                    x=time_df['MonthDate'],  # Asse X: mesi
                    y=time_df['Posts'],  # Asse Y: numero post
                    name='Posts'  # Nome nella legenda
                )
            ])
            fig_time.update_layout(
                xaxis_title="Date",  # Etichetta asse X
                yaxis_title="Number of Posts",  # Etichetta asse Y
                template="plotly_white"  # Tema bianco
            )
            st.plotly_chart(fig_time, use_container_width=True)  # Mostra il grafico
    
    # ==========================================================================
    # TAB 2: RANKINGS & RISK
    # Classifiche degli utenti più attivi e più a rischio
    # ==========================================================================
    with tab2:
        # Due colonne affiancate per le due classifiche
        col_stats_1, col_stats_2 = st.columns(2)
        
        # --- Colonna Sinistra: Top 10 Utenti per Attività ---
        with col_stats_1:
            st.markdown("### 📝 Top 10 Users by Activity")
            if top_activity is not None:
                # Mostra tabella con User_ID come indice
                st.table(top_activity.set_index("User_ID"))
        
        # --- Colonna Destra: Top 10 Utenti per Rischio (EMA) ---
        with col_stats_2:
            st.markdown("### ⚠️ Top 10 Users by Risk (EMA)")
            st.caption("Ranking based on latest Risk Score (Half-life: 15 days).")
            
            # Costruisce il percorso cache per il ranking di rischio
            cache_filename = f"global_rank_h15.json"  # File cache con half-life 15
            cache_path = os.path.join(data.GLOBAL_CACHE_DIR, cache_filename)
            
            # Prova a caricare il ranking dalla cache
            cached_df = None
            if os.path.exists(cache_path):
                try:
                    cached_df = pd.read_json(cache_path)  # Carica dalla cache
                except:
                    pass  # Ignora errori di lettura
            
            if cached_df is not None:
                # Se la cache esiste, mostra i primi 10 utenti
                disp_df = cached_df.head(10).copy()  # Prende i primi 10
                # Formatta le colonne numeriche con 4 decimali
                cols_to_fmt = ["Current Risk", "Avg Risk (EMA)", "Peak Risk"]
                for c in cols_to_fmt:
                    if c in disp_df.columns:
                        disp_df[c] = disp_df[c].astype(float).map("{:.4f}".format)
                # Mostra la tabella con User ID come indice
                st.table(disp_df.set_index("User ID"))
            else:
                # Se la cache non esiste, mostra pulsante per calcolare
                if st.button("Calculate Risk Rankings"):
                    with st.spinner("Calculating risk scores (Parallelized with Spark)..."):
                        # Calcola il ranking usando EMA con half-life 15 giorni
                        new_df = general_statistics.compute_risk_rankings(df, half_life=15)
                        if new_df is not None:
                            st.rerun()  # Ricarica la pagina per mostrare i risultati
                else:
                    st.info("Rankings not cached. Click to calculate.")
    
    # ==========================================================================
    # TAB 3: NUOVE ANALYTICS
    # Analisi aggiuntive calcolate con Spark SQL
    # ==========================================================================
    with tab3:
        # --- Sezione Attività per Giorno della Settimana ---
        st.markdown("### 📅 Attività per Giorno della Settimana")
        if weekday_df is not None and not weekday_df.empty:
            # Crea grafico a barre con i post per ogni giorno della settimana
            fig_wd = go.Figure(data=[
                go.Bar(
                    x=weekday_df['DayName'],  # Asse X: nomi dei giorni (Lun, Mar, ...)
                    y=weekday_df['Posts'],  # Asse Y: numero di post
                    name='Posts',  # Nome nella legenda
                    marker_color='steelblue'  # Colore delle barre: blu acciaio
                )
            ])
            fig_wd.update_layout(
                xaxis_title="Giorno",  # Etichetta asse X
                yaxis_title="Numero Post",  # Etichetta asse Y
                template="plotly_white"  # Tema bianco
            )
            st.plotly_chart(fig_wd, use_container_width=True)  # Mostra il grafico
        else:
            # Messaggio se i dati non sono ancora stati calcolati
            st.info("Dati non disponibili. Clicca 'Recalculate' per generarli.")
        
        st.markdown("---")  # Separatore
        
        # --- Sezione Statistiche Lunghezza Post ---
        st.markdown("### 📏 Statistiche Lunghezza Post")
        if length_df is not None and not length_df.empty:
            # Mostra media, minimo e massimo in 3 colonne
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Media Caratteri", f"{length_df['Avg_Length'].iloc[0]:.0f}")  # Media arrotondata
            c2.metric("📉 Min Caratteri", f"{length_df['Min_Length'].iloc[0]}")  # Post più corto
            c3.metric("📈 Max Caratteri", f"{length_df['Max_Length'].iloc[0]}")  # Post più lungo
        else:
            st.info("Dati non disponibili.")
        
        st.markdown("---")  # Separatore
        
        # --- Sezione Correlazione Attività-Rischio ---
        st.markdown("### 🔗 Correlazione Attività-Rischio (Top 20)")
        if correlation_df is not None and not correlation_df.empty:
            # Mostra la tabella dati completa
            st.dataframe(correlation_df, use_container_width=True)
            
            # Crea grafico a dispersione (scatter plot)
            fig_corr = go.Figure(data=[
                go.Scatter(
                    x=correlation_df['Post_Count'],  # Asse X: numero di post
                    y=correlation_df['Avg_Prob_Severe_Depressed'],  # Asse Y: media prob. severa
                    mode='markers',  # Solo punti, senza linee
                    marker=dict(size=10, color='darkred'),  # Punti grandi e rosso scuro
                    text=correlation_df['Subject_ID'],  # Testo hover: ID utente
                    hoverinfo='text+x+y'  # Mostra ID, X e Y al passaggio del mouse
                )
            ])
            fig_corr.update_layout(
                xaxis_title="Numero Post",  # Etichetta asse X
                yaxis_title="Avg Prob Severe Depressed",  # Etichetta asse Y
                title="Scatter: Attività vs Rischio",  # Titolo del grafico
                template="plotly_white"  # Tema bianco
            )
            st.plotly_chart(fig_corr, use_container_width=True)  # Mostra il grafico
        else:
            st.info("Dati non disponibili.")
    
    # ==========================================================================
    # TAB 4: VALUTAZIONE GPT
    # Mostra i risultati della valutazione qualitativa dei report con LLM
    # ==========================================================================
    with tab4:
        st.caption("Average scores from blinded Trajectory vs Base evaluations.")
        
        # Carica le statistiche aggregate delle valutazioni GPT dalla cache
        agg_data = gpt_evaluator.get_aggregate_stats()
        
        if agg_data:
            # Mostra il numero totale di valutazioni effettuate
            st.caption(f"Aggregated statistics from **{agg_data['total_evals']}** evaluations.")
            
            # Mostra le preferenze (vittorie) in 3 colonne
            c1, c2, c3 = st.columns(3)
            prefs = agg_data['preferences']  # Dizionario con conteggio preferenze
            c1.metric("Wins (Base)", prefs['Base'])  # Vittorie del report Base
            c2.metric("Wins (Trajectory)", prefs['Trajectory'])  # Vittorie del report Trajectory
            c3.metric("Ties", prefs['Tie'])  # Pareggi tra i due report
            
            # Mostra la tabella con i punteggi dettagliati per criterio
            st.table(agg_data['df'])
            
            # --- Sezione Riepilogo Qualitativo AI ---
            st.markdown("### 🧠 Qualitative AI Summary")
            st.caption("Aggregated insights generated by LLM across all evaluations.")
            
            # Carica il riepilogo qualitativo dalla cache
            summary_data = gpt_evaluator.load_qualitative_summary()
            
            if summary_data:
                # Mostra le giustificazioni per ogni criterio di valutazione
                justifications = summary_data.get("Criterion_Justifications", {})
                for k, v in justifications.items():
                    # Sostituisce gli underscore con spazi per leggibilità
                    st.info(f"**{k.replace('_', ' ')}**: {v}")
            
            # Pulsante per generare/rigenerare l'analisi qualitativa (richiede API key)
            if api_key:
                # Etichetta dinamica: "Regenerate" se esiste già, "Generate" se no
                btn_label = "Regenerate Qualitative Analysis" if summary_data else "Generate Qualitative Analysis"
                if st.button(btn_label):
                    with st.spinner("Synthesizing qualitative summary..."):
                        # Chiama l'API OpenAI per generare il riepilogo
                        new_summary = gpt_evaluator.generate_qualitative_summary(api_key)
                        if new_summary:
                            st.success("Summary generated!")
                            st.rerun()  # Ricarica per mostrare il risultato
            elif not summary_data:
                # Avviso se manca la chiave API e non c'è un riepilogo precedente
                st.warning("API Key required to generate qualitative summary.")
        else:
            # Messaggio se non ci sono valutazioni GPT in cache
            st.info("No evaluation data found in cache.")
