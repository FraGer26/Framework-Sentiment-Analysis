import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import data
import ema
import gpt_evaluator

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
        .getOrCreate()

def render_dataset_statistics(df, api_key):
    st.subheader("🌍 Global Dataset Statistics (Powered by Spark SQL)")
    
    # 0. Convert Payload to Spark
    # Only if it's substantial, otherwise this adds overhead. 
    # But for the assignment "Convert to Spark", we do it.
    spark = get_spark_session()
    
    # Clean column names for Spark compatibility (no spaces ideally, but we'll quote them)
    # df columns: ['Subject ID', 'Chunk', 'Date', 'Title', 'Info', 'Text', 'Label_User', ...]
    # We force string conversion for object types to ensure Arrow compatibility
    df_clean = df.copy()
    if 'Date' in df_clean.columns:
        df_clean['Date'] = df_clean['Date'].astype(str) # Pass as string, cast in Spark
    
    # Create Spark DF
    sdf = spark.createDataFrame(df_clean)
    sdf = sdf.withColumn("Date", to_date(col("Date"))) # Cast back to Date
    
    # Create Temp View for SQL access
    sdf.createOrReplaceTempView("reddit_posts")
    
    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 General Overview", "🏆 Rankings & Risk", "⚖️ GPT Evaluation"])
    
    with tab1:
        # 1. User Metrics (Spark SQL)
        # Query: Distinct Users, Total Posts
        metrics_df = spark.sql(queries.get_user_metrics_query("reddit_posts")).toPandas()
        
        num_users = metrics_df['num_users'][0]
        total_posts = metrics_df['total_posts'][0]
        avg_posts_user = total_posts / num_users if num_users > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Users", f"{num_users}")
        col2.metric("Total Posts", f"{total_posts}")
        col3.metric("Avg Posts/User", f"{avg_posts_user:.1f}")
        
        st.markdown("---")
        
        # 2. Depression Probability Distribution (Spark SQL)
        st.markdown("### Depression Probability (Dataset-wide)")
        
        has_dep_cols = "Prob_Severe_Depressed" in df.columns and "Prob_Moderate_Depressed" in df.columns
        
        if has_dep_cols:
            # Aggregate Averages via Spark
            avgs_df = spark.sql(queries.get_depression_averages_query("reddit_posts")).toPandas()
            
            avg_severe = avgs_df['avg_severe'][0]
            avg_moderate = avgs_df['avg_moderate'][0]
            avg_none = 1.0 - (avg_severe + avg_moderate)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Severe Score", f"{avg_severe:.4f}")
            c2.metric("Avg Moderate Score", f"{avg_moderate:.4f}")
            c3.metric("Avg Non-Depressed (Est.)", f"{avg_none:.4f}")
            
            # Histogram
            # NOTE: For Histograms, we usually need the raw data. 
            # If data is huge, we should compute bin counts in Spark.
            # Assuming data fits in memory for Plotly (since input `df` was Pandas), we use `df` directly for plotting to save code complexity.
            # Pure Spark approach would be: `sdf.select("Prob_Severe_Depressed").rdd.histogram(buckets)` but Plotly handles it better.
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df["Prob_Severe_Depressed"], name='Severe', opacity=0.75))
            fig_hist.add_trace(go.Histogram(x=df["Prob_Moderate_Depressed"], name='Moderate', opacity=0.75))
            fig_hist.update_layout(barmode='overlay', title="Distribution of Risk Scores", xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Depression probability columns not found.")

        st.markdown("---")
        
        # 3. Activity Over Time (Spark SQL)
        st.markdown("### 📈 Posts Volume Over Time")
        if 'Date' in df.columns:
            # Group by Month using Spark SQL
            time_df = spark.sql(queries.get_posts_over_time_query("reddit_posts")).toPandas()
            
            fig_time = go.Figure(data=[go.Bar(x=time_df['MonthDate'], y=time_df['Posts'], name='Posts')])
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", template="plotly_white")
            st.plotly_chart(fig_time, use_container_width=True)
            
    with tab2:
        col_stats_1, col_stats_2 = st.columns(2)
        
        with col_stats_1:
             st.markdown("### 📝 Top 10 Users by Activity")
             # Spark SQL Ranking
             top_activity = spark.sql(queries.get_top_active_users_query("reddit_posts")).toPandas()
             
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
             
             # --- Spark Parallelized Calculation ---
             def perform_calculation_spark():
                 with st.spinner("Calculating risk scores (Parallelized with Spark)..."):
                    
                     # Define the Pandas UDF Schema
                     # Input: DataFrame for one user
                     # Output: DataFrame with 1 row (User ID, Current, Avg, Peak)
                     result_schema = StructType([
                         StructField("User ID", StringType(), True),
                         StructField("Current Risk", FloatType(), True),
                         StructField("Avg Risk (EMA)", FloatType(), True),
                         StructField("Peak Risk", FloatType(), True)
                     ])
                     
                     def calculate_risk_udf(pdf):
                         # pdf is a pandas dataframe for a single user
                         if pdf.empty: 
                             return pd.DataFrame(columns=["User ID", "Current Risk", "Avg Risk (EMA)", "Peak Risk"])
                             
                         uid = str(pdf["Subject ID"].iloc[0])
                         
                         # Call existing logic (which handles internal caching per user)
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

                     # Apply to Spark DF
                     results_sdf = sdf.groupBy("Subject ID").applyInPandas(calculate_risk_udf, schema=result_schema)
                     
                     # Sort and Take Top 20 (Action)
                     # We move sorting to Spark side before collecting
                     top_risky = results_sdf.orderBy(desc("Current Risk")).limit(20).toPandas()
                     
                     # Save to disk
                     top_risky.to_json(cache_path)
                     return top_risky
     
             # Logic: If cached, show. If button, calc.
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
