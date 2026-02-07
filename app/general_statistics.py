import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import data
import ema
import gpt_evaluator
import queries

# --- Spark Integration ---
# We initialize Spark strictly when needed to avoid overhead if not used.
# Ideally this should be a singleton or session-state object.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, to_date, count, avg, desc
from pyspark.sql.types import StructType, StructField, StringType, FloatType

@st.cache_resource
def get_spark_session():
    # Use 8GB driver memory to handle "pd.toPandas()" if needed, though we avoid it for big data
    # "local[*]" uses all cores
    return SparkSession.builder \
        .appName("RedditAnalyticsApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.heartbeatInterval", "100s") \
        .config("spark.network.timeout", "300s") \
        .getOrCreate()

def render_dataset_statistics(df, api_key):
    st.subheader("🌍 Global Dataset Statistics (Powered by Spark SQL)")
    
    # --- Cache Paths ---
    cache_metrics_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_metrics.json")
    cache_avgs_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_avgs.json")
    cache_time_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_time.json")
    cache_top_path = os.path.join(data.GLOBAL_CACHE_DIR, "global_top_users.json")
    
    # Check if caches exist
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
    
    if caches_exist and not recalc:
        try:
            metrics_df = pd.read_json(cache_metrics_path)
            avgs_df = pd.read_json(cache_avgs_path)
            time_df = pd.read_json(cache_time_path).sort_values("MonthDate") # JSON might lose order
            top_activity = pd.read_json(cache_top_path)
            st.success("Loaded statistics from cache.")
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Recalculating...")
            caches_exist = False

    if not caches_exist or recalc:
        with st.spinner("Initializing Spark and calculating statistics... (This may take a moment)"):
             # 0. Convert Payload to Spark
             spark = get_spark_session()
             
             # Clean column names for Spark compatibility
             df_clean = df.copy()
             if 'Date' in df_clean.columns:
                 df_clean['Date'] = df_clean['Date'].astype(str)
             
             # Create Spark DF
             sdf = spark.createDataFrame(df_clean)
             sdf = sdf.withColumn("Date", to_date(col("Date")))
             
             # 1. User Metrics
             metrics_df = queries.get_user_metrics_df(sdf).toPandas()
             metrics_df.to_json(cache_metrics_path)
             
             # 2. Depression Averages
             if "Prob_Severe_Depressed" in df.columns:
                 avgs_df = queries.get_depression_averages_df(sdf).toPandas()
                 avgs_df.to_json(cache_avgs_path)
             
             # 3. Posts Over Time
             if 'Date' in df.columns:
                 time_df = queries.get_posts_over_time_df(sdf).toPandas()
                 time_df.to_json(cache_time_path)
                 
             # 4. Top Active Users
             top_activity = queries.get_top_active_users_df(sdf).toPandas()
             top_activity.to_json(cache_top_path)
             
             st.success("Statistics calculated and cached.")
             # Rerun to pick up cached values cleanly if needed, or just proceed
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 General Overview", "🏆 Rankings & Risk", "⚖️ GPT Evaluation"])
    
    with tab1:
        # 1. User Metrics
        if metrics_df is not None and not metrics_df.empty:
            num_users = metrics_df['num_users'][0]
            total_posts = metrics_df['total_posts'][0]
            avg_posts_user = total_posts / num_users if num_users > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", f"{num_users}")
            col2.metric("Total Posts", f"{total_posts}")
            col3.metric("Avg Posts/User", f"{avg_posts_user:.1f}")
        
        st.markdown("---")
        
        # 2. Depression Probability Distribution
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
            
            # Histogram (Using raw DF for plotting as before, fast enough)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df["Prob_Severe_Depressed"], name='Severe', opacity=0.75))
            fig_hist.add_trace(go.Histogram(x=df["Prob_Moderate_Depressed"], name='Moderate', opacity=0.75))
            fig_hist.update_layout(barmode='overlay', title="Distribution of Risk Scores", xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Depression probability columns not found.")

        st.markdown("---")
        
        # 3. Activity Over Time
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
             # 4. Top Risky Users (EMA Based) - Cached
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
             
             # --- Spark Parallelized Calculation (Only if needed) ---
             def perform_calculation_spark():
                 # Re-init spark here if needed, but optimally we pass 'sdf' if we had it.
                 # Since 'sdf' might not be created if we hit cache above, we need to ensure spark is ready.
                 spark = get_spark_session()
                 # Re-create SDF for this specific calculation if not present. 
                 # This is a bit inefficient if we just calculated above, but cleaner for separation.
                 # To optimize, we could store 'sdf' in session state or re-use logic.
                 # For now, let's just recreate logic.
                 df_clean_local = df.copy()
                 if 'Date' in df_clean_local.columns:
                     df_clean_local['Date'] = df_clean_local['Date'].astype(str)
                 sdf_local = spark.createDataFrame(df_clean_local)
                 sdf_local = sdf_local.withColumn("Date", to_date(col("Date")))

                 with st.spinner("Calculating risk scores (Parallelized with Spark)..."):
                    
                     # Define the Pandas UDF Schema
                     result_schema = StructType([
                         StructField("User ID", StringType(), True),
                         StructField("Current Risk", FloatType(), True),
                         StructField("Avg Risk (EMA)", FloatType(), True),
                         StructField("Peak Risk", FloatType(), True)
                     ])
                     
                     def calculate_risk_udf(pdf):
                         if pdf.empty: 
                             return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])
                         uid = str(pdf["Subject ID"].iloc[0])
                         r_series = ema.calculate_risk_score(pdf, half_life=15)
                         if r_series.empty:
                             return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])
                         curr_risk = float(r_series.iloc[-1])
                         avg_risk = float(r_series.mean())
                         max_risk = float(r_series.max())
                         return pd.DataFrame([{
                             "User ID": uid,
                             "Current Risk": curr_risk,
                             "Avg Risk (EMA)": avg_risk,
                             "Peak Risk": max_risk
                         }])

                     results_sdf = sdf_local.groupBy("Subject ID").applyInPandas(calculate_risk_udf, schema=result_schema)
                     top_risky = results_sdf.orderBy(desc("Current Risk")).limit(20).toPandas()
                     top_risky.to_json(cache_path)
                     return top_risky
     
             if cached_df is not None:
                  disp_df = cached_df.head(10).copy()
                  cols_to_fmt = ["Current Risk", "Avg Risk (EMA)", "Peak Risk"]
                  for c in cols_to_fmt:
                      if c in disp_df.columns:
                          disp_df[c] = disp_df[c].astype(float).map("{:.4f}".format)
                  st.table(disp_df.set_index("User ID"))
             else:
                 if st.button("Calculate Risk Rankings"):
                     new_df = perform_calculation_spark()
                     if new_df is not None:
                          st.rerun()
                 else:
                     st.info("Rankings not cached. Click to calculate.")
    
    with tab3:
        # 5. GPT Evaluator (No Spark needed, small data)
        st.caption("Average scores from blinded Trajectory vs Base evaluations.")
        
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
