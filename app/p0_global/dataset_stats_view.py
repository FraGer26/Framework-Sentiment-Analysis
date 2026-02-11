import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from p0_global import data, general_statistics
from p3_llm_judge import gpt_evaluator

def render_dataset_statistics(df, api_key):
    st.subheader("🌍 Global Dataset Statistics")
    
    # --- Percorsi Cache ---
    # Accesso dal modulo data (costanti centralizzate presunte)
    cache_metrics_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_metrics.json")
    cache_avgs_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_avgs.json")
    cache_time_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_time.json")
    cache_top_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_top_users.json")
    cache_weekday_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_weekday.json")
    cache_length_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_length.json")
    cache_correlation_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_correlation.json")
    
    # Controlla esistenza cache base
    caches_exist = (
        os.path.exists(cache_metrics_path) and 
        os.path.exists(cache_avgs_path) and 
        os.path.exists(cache_time_path) and 
        os.path.exists(cache_top_path)
    )
    
    recalc = st.button("🔄 Recalculate Global Statistics")
    
    metrics_df = None
    avgs_df = None
    time_df = None
    top_activity = None
    weekday_df = None
    length_df = None
    correlation_df = None
    
    if caches_exist and not recalc:
        try:
            metrics_df = pd.read_json(cache_metrics_path)
            avgs_df = pd.read_json(cache_avgs_path)
            time_df = pd.read_json(cache_time_path).sort_values("MonthDate")
            top_activity = pd.read_json(cache_top_path)
            # Carica nuove analytics dalla cache
            if os.path.exists(cache_weekday_path):
                weekday_df = pd.read_json(cache_weekday_path)
            if os.path.exists(cache_length_path):
                length_df = pd.read_json(cache_length_path)
            if os.path.exists(cache_correlation_path):
                correlation_df = pd.read_json(cache_correlation_path)
            st.success("Loaded statistics from cache.")
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Recalculating...")
            caches_exist = False

    if not caches_exist or recalc:
        with st.spinner("Initializing Spark and calculating statistics... (This may take a moment)"):
             metrics_df, avgs_df, time_df, top_activity, weekday_df, length_df, correlation_df = general_statistics.compute_global_stats(df)
             st.success("Statistics calculated and cached.")
    
    # Tab per organizzazione
    tab1, tab2, tab3, tab4 = st.tabs(["📊 General Overview", "🏆 Rankings & Risk", "📅 New Analytics", "⚖️ GPT Evaluation"])
    
    with tab1:
        # 1. Metriche Utente
        if metrics_df is not None and not metrics_df.empty:
            num_users = metrics_df['num_users'][0]
            total_posts = metrics_df['total_posts'][0]
            avg_posts_user = total_posts / num_users if num_users > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", f"{num_users}")
            col2.metric("Total Posts", f"{total_posts}")
            col3.metric("Avg Posts/User", f"{avg_posts_user:.1f}")
        
        st.markdown("---")
        
        # 2. Distribuzione Probabilità Depressione
        st.markdown("### Depression Probability (Dataset-wide)")
        
        has_dep_cols = "Prob_Severe_Depressed" in df.columns and "Prob_Moderate_Depressed" in df.columns
        
        if has_dep_cols:
            if avgs_df is not None and not avgs_df.empty:
                avg_severe = avgs_df['avg_severe'][0]
                avg_moderate = avgs_df['avg_moderate'][0]
                avg_none = 1.0 - (avg_severe + avg_moderate)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Severe Score", f"{avg_severe:.4f}")
                c2.metric("Avg Moderate Score", f"{avg_moderate:.4f}")
                c3.metric("Avg Non-Depressed (Est.)", f"{avg_none:.4f}")
            
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df["Prob_Severe_Depressed"], name='Severe', opacity=0.75))
            fig_hist.add_trace(go.Histogram(x=df["Prob_Moderate_Depressed"], name='Moderate', opacity=0.75))
            fig_hist.update_layout(barmode='overlay', title="Distribution of Risk Scores", xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Depression probability columns not found.")
            
        st.markdown("---")
        
        # 3. Attività nel Tempo
        st.markdown("### 📈 Posts Volume Over Time")
        if 'Date' in df.columns and time_df is not None:
            fig_time = go.Figure(data=[go.Bar(x=time_df['MonthDate'], y=time_df['Posts'], name='Posts')])
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", template="plotly_white")
            st.plotly_chart(fig_time, use_container_width=True)
            
    with tab2:
        col_stats_1, col_stats_2 = st.columns(2)
        
        with col_stats_1:
             st.markdown("### 📝 Top 10 Users by Activity")
             if top_activity is not None:
                 st.table(top_activity.set_index("User_ID"))
        
        with col_stats_2:
             # 4. Utenti più a Rischio (Basato su EMA) - Cache
             st.markdown("### ⚠️ Top 10 Users by Risk (EMA)")
             st.caption("Ranking based on latest Risk Score (Half-life: 15 days).")
             
             cache_filename = f"global_rank_h15.json" 
             cache_path = os.path.join(data.GLOBAL_CACHE_DIR, cache_filename)
             
             cached_df = None
             if os.path.exists(cache_path):
                 try:
                      cached_df = pd.read_json(cache_path)
                 except:
                      pass
             
             if cached_df is not None:
                  disp_df = cached_df.head(10).copy()
                  cols_to_fmt = ["Current Risk", "Avg Risk (EMA)", "Peak Risk"]
                  for c in cols_to_fmt:
                      if c in disp_df.columns:
                          disp_df[c] = disp_df[c].astype(float).map("{:.4f}".format)
                  st.table(disp_df.set_index("User ID"))
             else:
                 if st.button("Calculate Risk Rankings"):
                     with st.spinner("Calculating risk scores (Parallelized with Spark)..."):
                         new_df = general_statistics.compute_risk_rankings(df, half_life=15)
                         if new_df is not None:
                              st.rerun()
                 else:
                     st.info("Rankings not cached. Click to calculate.")
    
    with tab3:
        st.markdown("### 📅 Attività per Giorno della Settimana")
        if weekday_df is not None and not weekday_df.empty:
            fig_wd = go.Figure(data=[
                go.Bar(x=weekday_df['DayName'], y=weekday_df['Posts'], name='Posts', marker_color='steelblue')
            ])
            fig_wd.update_layout(xaxis_title="Giorno", yaxis_title="Numero Post", template="plotly_white")
            st.plotly_chart(fig_wd, use_container_width=True)
        else:
            st.info("Dati non disponibili. Clicca 'Recalculate' per generarli.")
        
        st.markdown("---")
        st.markdown("### 📏 Statistiche Lunghezza Post")
        if length_df is not None and not length_df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Media Caratteri", f"{length_df['Avg_Length'].iloc[0]:.0f}")
            c2.metric("📉 Min Caratteri", f"{length_df['Min_Length'].iloc[0]}")
            c3.metric("📈 Max Caratteri", f"{length_df['Max_Length'].iloc[0]}")
        else:
            st.info("Dati non disponibili.")
        
        st.markdown("---")
        st.markdown("### 🔗 Correlazione Attività-Rischio (Top 20)")
        if correlation_df is not None and not correlation_df.empty:
            st.dataframe(correlation_df, use_container_width=True)
            fig_corr = go.Figure(data=[
                go.Scatter(x=correlation_df['Post_Count'], y=correlation_df['Avg_Prob_Severe_Depressed'], 
                           mode='markers', marker=dict(size=10, color='darkred'),
                           text=correlation_df['Subject_ID'], hoverinfo='text+x+y')
            ])
            fig_corr.update_layout(xaxis_title="Numero Post", yaxis_title="Avg Prob Severe Depressed", 
                                   title="Scatter: Attività vs Rischio", template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Dati non disponibili.")
    
    with tab4:
        # 5. Valutatore GPT
        st.caption("Average scores from blinded Trajectory vs Base evaluations.")
        
        # Nota: gpt_evaluator potrebbe necessitare di Streamlit per caching o chiamate dirette
        # Assumendo che gpt_evaluator abbia logica di caching interna o chiamiamo get_aggregate_stats
        agg_data = gpt_evaluator.get_aggregate_stats()
        
        if agg_data:
            st.caption(f"Aggregated statistics from **{agg_data['total_evals']}** evaluations.")
            
            c1, c2, c3 = st.columns(3)
            prefs = agg_data['preferences']
            c1.metric("Wins (Base)", prefs['Base'])
            c2.metric("Wins (Trajectory)", prefs['Trajectory'])
            c3.metric("Ties", prefs['Tie'])
            
            st.table(agg_data['df'])
            
            st.markdown("### 🧠 Qualitative AI Summary")
            st.caption("Aggregated insights generated by LLM across all evaluations.")
            
            summary_data = gpt_evaluator.load_qualitative_summary()
            
            if summary_data:
                justifications = summary_data.get("Criterion_Justifications", {})
                for k, v in justifications.items():
                    st.info(f"**{k.replace('_', ' ')}**: {v}")
            
            if api_key:
                btn_label = "Regenerate Qualitative Analysis" if summary_data else "Generate Qualitative Analysis"
                if st.button(btn_label):
                    with st.spinner("Synthesizing qualitative summary..."):
                        new_summary = gpt_evaluator.generate_qualitative_summary(api_key)
                        if new_summary:
                            st.success("Summary generated!")
                            st.rerun()
            elif not summary_data:
                st.warning("API Key required to generate qualitative summary.")
        else:
            st.info("No evaluation data found in cache.")
