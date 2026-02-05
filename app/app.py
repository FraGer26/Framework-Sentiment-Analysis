import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import data
import ema
import segment
import json
import topic_coverage
import text_coverage
import re
import report_base
import report_trajectory
import clustering
import topic_model

# --- Configuration ---
st.set_page_config(
    page_title="Reddit User Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def render_global_overview(user_data):
    st.subheader("User Activity Timeline")
    # Simple Histogram of posts over time
    daily_counts = user_data.groupby(user_data["Date"].dt.date).size()
    fig_activity = go.Figure(data=[
        go.Bar(x=daily_counts.index, y=daily_counts.values, name="Posts")
    ])
    fig_activity.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", template="plotly_white")
    st.plotly_chart(fig_activity, use_container_width=True)
    
    st.subheader("Recent Posts")
    st.dataframe(user_data[["Date", "Text", "Prob_Severe_Depressed", "Prob_Moderate_Depressed"]].sort_values("Date", ascending=False).head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top Risk Posts")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### Top 5 Severely Depressed")
        top_severe = user_data.sort_values("Prob_Severe_Depressed", ascending=False).head(5)
        st.dataframe(top_severe[["Date", "Prob_Severe_Depressed", "Text"]], use_container_width=True, hide_index=True)
        
    with col_t2:
        st.markdown("#### Top 5 Moderately Depressed")
        top_moderate = user_data.sort_values("Prob_Moderate_Depressed", ascending=False).head(5)
        st.dataframe(top_moderate[["Date", "Prob_Moderate_Depressed", "Text"]], use_container_width=True, hide_index=True)

def render_risk_dashboard(user_data, risk_series, segments, half_life):
    st.subheader("Risk Score Evolution & Segmentation")
    
    if risk_series.empty:
        st.warning("Not enough data to calculate risk score.")
    else:
        # Plot
        fig = go.Figure()
        
        # 1. EMA Curve
        fig.add_trace(go.Scatter(
            x=risk_series.index, 
            y=risk_series.values,
            mode='lines',
            name='Risk Score (EMA)',
            line=dict(color='rgba(0, 100, 255, 0.3)', width=2),
            hoverinfo='x+y'
        ))
        
        # 2. Segments (Colored by Trend)
        # 2. Segments (Multicolor)
        if segments:
            # Vibrant palette
            palette = [
                '#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', 
                '#00FFFF', '#FF00FF', '#FFFF00', '#008080', '#A52A2A'
            ]
            
            for i, seg in enumerate(segments):
                start_val = seg['start_val']
                end_val = seg['end_val']
                
                # Cycle through palette
                seg_color = palette[i % len(palette)]
                
                # Show in legend only for the first segment
                show_leg = True if i == 0 else False
                # Name it generic "Segments" if it's the first one, else use individual name but hidden
                trace_name = "Segments" if i == 0 else f'Segment {i+1}'
                
                fig.add_trace(go.Scatter(
                    x=[seg['start_date'], seg['end_date']],
                    y=[start_val, end_val],
                    mode='lines',  # Visual only
                    name=trace_name,
                    line=dict(color=seg_color, width=4),
                    showlegend=show_leg,
                    legendgroup="segments",
                    hoverinfo='skip' # Disable hover on colored segments to prevent duplicates
                ))
        
        # 3. Invisible Unified Trace for Clean Hover
        # This trace connects all segments but deduplicates the join points for hover purposes
        if segments:
            unified_x = []
            unified_y = []
            for i, seg in enumerate(segments):
                # Add start point only if it's the very first segment or if there's a gap/jump
                # Assuming continuous, we typically skip start if it matches previous end. 
                # Simplest robust way: if unified_x is empty, add start. Else if start != last, add start.
                if not unified_x:
                    unified_x.append(seg['start_date'])
                    unified_y.append(seg['start_val'])
                elif seg['start_date'] != unified_x[-1]:
                     unified_x.append(seg['start_date'])
                     unified_y.append(seg['start_val'])
                
                # Always add end point
                unified_x.append(seg['end_date'])
                unified_y.append(seg['end_val'])
                
            fig.add_trace(go.Scatter(
                x=unified_x,
                y=unified_y,
                mode='lines',
                name='Segment Trend',
                line=dict(width=0), # Invisible line
                opacity=0,
                showlegend=False,
                hovertemplate="%{y:.4f}<extra>Segment Value</extra>", # Clean label
                hoverinfo='y' # or just rely on template
            ))

        breakpoints = []
        for seg in segments:
            breakpoints.append({'Date': seg['end_date'], 'Score': seg['end_val']})
        
        # 4. Highlight Breakpoints (Visual only)
        if breakpoints:
            bp_df = pd.DataFrame(breakpoints)
            fig.add_trace(go.Scatter(
                x=bp_df['Date'],
                y=bp_df['Score'],
                mode='markers',
                name='Breakpoints',
                marker=dict(symbol='x', size=10, color='black'),
                hoverinfo='skip' # Disable hover to avoid double-label with Segment Trend
            ))


        fig.update_layout(
            title=f"Risk Evolution (Half-Life: {half_life} days)",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Breakpoints Table
        st.subheader("Identified Breakpoints (Trend Changes)")
        if breakpoints:
            bp_df = pd.DataFrame(breakpoints)
            bp_df["Date"] = pd.to_datetime(bp_df["Date"]).dt.strftime("%Y-%m-%d")
            st.table(bp_df.set_index("Date"))

def render_trajectory_section(selected_user, user_data, segments, api_key):
    # Subheader removed as per user request (redundant)
    
    base_data = report_base.load_base_report(selected_user)
    traj_data = report_trajectory.load_trajectory_report(selected_user)
    is_traj_cached = (base_data is not None) and (traj_data is not None)
    
    if is_traj_cached:
        st.success("Narrative analysis available in cache. Loading...")
        with st.spinner("Loading narrative from cache..."):
            # Load both parts
            b_data, _ = report_base.generate_base_report(selected_user, user_data.copy(), api_key)
            t_data, _ = report_trajectory.generate_trajectory_report(selected_user, user_data.copy(), segments, api_key)
            combined = {**b_data, **t_data} if b_data and t_data else None
            
            if combined:
                 # Base Report
                 base_text = combined.get('base_analysis', 'N/A')
                 wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                 
                 st.markdown("### 📝 Overall Narrative (Base)")
                 st.write(base_text)
                 st.caption(f"Word Count: {wc_base}")
                 
                 # Trajectory Report Metrics
                 traj_summary = combined.get('trajectory_summary', 'N/A')
                 phases = combined.get('phases', [])
                 
                 wc_summary = len(re.findall(r"\b\w+\b", traj_summary)) if traj_summary and traj_summary != 'N/A' else 0
                 wc_phases = sum([len(re.findall(r"\b\w+\b", p['narrative'])) for p in phases])
                 wc_total_traj = wc_summary + wc_phases
                 
                 st.markdown("### 🔄 Trajectory Summary")
                 st.write(traj_summary) # Changed from info to write for consistency, or keep info? User didn't specify. Keeping info or write? Original was info. Let's keep info for summary if user likes it, but usually write is cleaner for large text. Let's stick to st.info as per original but maybe caption underneath.
                 st.caption(f"Total Trajectory Word Count (Summary + Phases): {wc_total_traj}")
                 
                 st.markdown("### 📅 Phase-by-Phase Evolution")
                 for phase in combined.get('phases', []):
                     with st.expander(f"Phase {phase['phase_num']}: {phase['start_date']} to {phase['end_date']} (Delta: {phase['delta']:.4f})"):
                         st.write(phase['narrative'])
    else:
        if st.button("Generate Narrative Analysis"):
            if not api_key:
                 st.error("Please provide an OpenAI API Key in the sidebar to generate narrative.")
            else:
                with st.spinner("Analyzing trajectory..."):
                    if not segments:
                         st.warning("No segments available to analyze.")
                    else:
                        b_data, _ = report_base.generate_base_report(selected_user, user_data.copy(), api_key)
                        t_data, _ = report_trajectory.generate_trajectory_report(selected_user, user_data.copy(), segments, api_key)
                        combined = {**b_data, **t_data} if b_data and t_data else None
                        
                        if combined:
                            st.info("Analysis generated from API and saved to cache.")
                            
                            # Base Report
                            base_text = combined.get('base_analysis', 'N/A')
                            wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                            
                            st.markdown("### 📝 Overall Narrative (Base)")
                            st.write(base_text)
                            st.caption(f"Word Count: {wc_base}")
                            
                            # Trajectory Report Metrics
                            traj_summary = combined.get('trajectory_summary', 'N/A')
                            phases = combined.get('phases', [])
                            
                            wc_summary = len(re.findall(r"\b\w+\b", traj_summary)) if traj_summary and traj_summary != 'N/A' else 0
                            wc_phases = sum([len(re.findall(r"\b\w+\b", p['narrative'])) for p in phases])
                            wc_total_traj = wc_summary + wc_phases
                            
                            st.markdown("### 🔄 Trajectory Summary")
                            st.info(traj_summary)
                            st.caption(f"Total Trajectory Word Count (Summary + Phases): {wc_total_traj}")
                            
                            st.markdown("### 📅 Phase-by-Phase Evolution")
                            for phase in combined.get('phases', []):
                                with st.expander(f"Phase {phase['phase_num']}: {phase['start_date']} to {phase['end_date']} (Delta: {phase['delta']:.4f})"):
                                    st.write(phase['narrative'])
                        else:
                            st.error("Failed to generate analysis.")

def render_topic_analysis(selected_user, user_data, api_key):
    # Subheader removed

    
    # 1. Source Selection
    topic_source = st.selectbox(
        "Select Topic Source", 
        ["Raw Posts", "Narrative Base", "Narrative Trajectory"],
        key="topic_source_select"
    )
    
    # Determine text and source_key
    text_to_analyze = ""
    source_key = "posts" # Default
    
    if topic_source == "Raw Posts":
        text_to_analyze = "\n".join(user_data["Text"].astype(str))
        source_key = "posts"
    else:
        # Load Narrative
        # Load Narrative from split cache
        base_data = report_base.load_base_report(selected_user)
        traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
        
        narrative_base = ""
        narrative_traj = ""
        
        if base_data:
            narrative_base = base_data.get('base_analysis', '')
        if traj_data_loaded:
            narrative_traj = traj_data_loaded.get('trajectory_summary', '')
        
        if topic_source == "Narrative Base":
            text_to_analyze = narrative_base
            source_key = "narrative_base"
        elif topic_source == "Narrative Trajectory":
            text_to_analyze = narrative_traj
            source_key = "narrative_traj"
    
    if not text_to_analyze:
        st.warning(f"No text available for {topic_source}. Please run Narrative Analysis or check data.")
        return

    # Check cache for topics
    cache_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{source_key}_topics.json")
    is_cached = os.path.exists(cache_path)
    
    topics_data = None
    
    if is_cached:
        st.success(f"Topic analysis for **{topic_source}** available in cache.")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)
        except:
            pass
    else:
        st.info(f"No cached topics for **{topic_source}**.")
        if st.button(f"Extract Topics ({topic_source})"):
             if not api_key:
                 st.error("API Key required.")
             else:
                 with st.spinner("Extracting topics..."):
                     # Call topic_model.extract_topics (Correct signature: user_id, text, api_key, source_type)
                     topics_data, _ = topic_model.extract_topics(selected_user, text_to_analyze, api_key, source_type=source_key)
                     if topics_data:
                         st.success("Topics extracted!")
                         st.rerun()
    
    if topics_data:
        col_list1, col_list2, col_list3 = st.columns(3)
        with col_list1:
            st.markdown("#### Positive Topics")
            for t in topics_data.get("positivetopics", []): st.write(f"- {t}")
        with col_list2:
           st.markdown("#### Neutral Topics")
           for t in topics_data.get("neutraltopics", []): st.write(f"- {t}")
        with col_list3:
           st.markdown("#### Negative Topics")
           for t in topics_data.get("negativetopics", []): st.write(f"- {t}")

def render_topic_coverage(selected_user):
    # Subheader removed

    st.caption("Calculate how well the Narrative Topics cover the Raw Ground Truth Topics.")
    
    # 1. Load Raw Topics (Ground Truth)
    raw_topics = []
    rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
    if os.path.exists(rt_path):
        try:
            with open(rt_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
            
    if not raw_topics:
        st.warning("Ground Truth (Raw Posts) topics not found. Please extract them in 'Topic Analysis' first.")
        return

    # 2. Select Candidate Source
    candidate_source = st.selectbox("Select Candidate Topics", ["Narrative Base", "Narrative Trajectory"], key="topic_cov_cand")
    cand_key = "narrative_base" if candidate_source == "Narrative Base" else "narrative_traj"
    
    # Load Candidate Topics
    cand_topics = []
    ct_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{cand_key}_topics.json")
    if os.path.exists(ct_path):
        try:
            with open(ct_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                cand_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
            
    if not cand_topics:
        st.warning(f"Candidate topics ({candidate_source}) not found. Please extract them in 'Topic Analysis' first.")
        return
        
    st.write(f"**Reference:** {len(raw_topics)} topics | **Candidate:** {len(cand_topics)} topics")
    
    cov_threshold = st.slider("Topic Similarity Threshold", 0.5, 1.0, 0.75, 0.05)
    
    # Check Cache
    is_cached = topic_coverage.check_cache(selected_user, cand_key, threshold=cov_threshold)
    metrics = None
    df_matches = None

    if is_cached and 'topic_metrics' not in st.session_state:
         # Note: We use a different session key or just local vars? 
         # The original code didn't use session state for topics coverage, it just showed it.
         # But if we want it to persist or be usable, local vars are fine if we render immediately.
         metrics, df_matches, _ = topic_coverage.load_cache(selected_user, cand_key, threshold=cov_threshold)
         if metrics:
             st.success("Topic coverage results loaded from cache.")

    if st.button("Calculate Topic Coverage") or (is_cached and metrics is None):
        if not metrics:
            with st.spinner("Calculating matching..."):
                metrics, df_matches, _ = topic_coverage.calculate_coverage_metrics(selected_user, cand_key, raw_topics, cand_topics, threshold=cov_threshold)
            
    if metrics:
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{metrics['precision']:.2f}")
                c2.metric("Recall", f"{metrics['recall']:.2f}")
                c3.metric("F1 Score", f"{metrics['f1']:.2f}")
                
                # Sensitivity Analysis (Line Chart)
                sens_data = topic_coverage.sensitivity_analysis(raw_topics, cand_topics)
                
                if sens_data:
                    import plotly.graph_objects as go
                    fig_sens = go.Figure()
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
                    
                    fig_sens.add_vline(x=cov_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
                    
                    fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
                    st.plotly_chart(fig_sens, use_container_width=True)
                else:
                    st.error("Could not generate sensitivity data.")

                st.write("#### Matches")
                # Show only matches
                st.dataframe(df_matches[df_matches['matched'] == True], use_container_width=True)
                
                st.write("#### Missed (Ref Topics NOT covered)")
                missed = df_matches[df_matches['matched'] == False]['full_topic'].tolist()
                st.write(missed)

def render_clustering_section(selected_user, user_data):
    # Subheader removed

    st.caption("Advanced clustering of posts using BERTopic.")
    
    if 'user_data' in locals() and not user_data.empty and "Text" in user_data.columns:
         texts_full = user_data["Text"].astype(str).tolist()
    else:
         texts_full = user_data["Text"].astype(str).tolist() if not user_data.empty else []

    if not texts_full:
        st.warning("No posts available for clustering.")
    else:
        cluster_col1, cluster_col2 = st.columns([1,3])
        with cluster_col1:
            cluster_source = st.selectbox("Source Text", ["Raw Posts"], key="cluster_source")
            current_sig = "umap_n40_c5_d0_s42__hdbscan_m35_predT" 
            
            is_cluster_cached = clustering.check_cache(selected_user, params_sig=current_sig)
            should_run_cluster = False
            
            if is_cluster_cached and 'cluster_results' not in st.session_state:
                cluster_results, from_cache_cluster = clustering.load_cache(selected_user, params_sig=current_sig)
                if from_cache_cluster:
                    st.success("✅ Clusters loaded from cache.")
                    st.session_state['cluster_results'] = cluster_results

            if 'cluster_results' in st.session_state:
                 cluster_results = st.session_state['cluster_results']
                 # Ensure we don't show the button if we have results? Or show "Re-run"?
                 # Just show success message above. 
            
            if not is_cluster_cached and 'cluster_results' not in st.session_state:
                 st.info("No cached clusters.")
                 if st.button("Run Clustering"):
                     should_run_cluster = True
            elif st.button("Re-Run Clustering"): # Optional re-run
                 should_run_cluster = True
        
        raw_topics = []
        rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
        if os.path.exists(rt_path):
            try:
                with open(rt_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
            except:
                pass
        
        if should_run_cluster:
            with st.spinner("Running BERTopic clustering..."):
                cluster_results = clustering.run_clustering(selected_user, texts_full)
                st.session_state['cluster_results'] = cluster_results
        
        if 'cluster_results' in locals() and cluster_results:
            st.write("### Clustering Results")
            topic_df = cluster_results["topic_info"]
            
            if raw_topics:
                with st.spinner("Mapping topics to Ground Truth..."):
                     topic_df = clustering.map_topics_to_ground_truth(topic_df, raw_topics)
            
            st.write("#### Topic Overview")
            st.dataframe(topic_df, use_container_width=True)
            
            st.write("#### Cluster Visualization")
            fig_cluster = clustering.visualize_clusters(cluster_results["vis_data"], topic_info_df=topic_df)
            if fig_cluster:
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Visualization data not available.")

def render_text_coverage(selected_user, user_data):
    # Subheader removed

    texts_full = user_data["Text"].astype(str).tolist()
    
    base_data = report_base.load_base_report(selected_user)
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
    
    narrative_base = ""
    narrative_traj = ""
    
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    if traj_data_loaded:
        narrative_traj = traj_data_loaded.get('trajectory_summary', '')

    coverage_source = st.selectbox("Compare Against", ["Narrative Base", "Narrative Trajectory"], key="cov_source")
    comparison_text = narrative_base if coverage_source == "Narrative Base" else narrative_traj
        
    if not comparison_text:
        st.warning("Selected narrative text is empty.")
    
    coverage_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # Determine source type key for cache
    metrics = None
    df_matches = None
    
    source_key = "base" if coverage_source == "Narrative Base" else "traj"
     
    # Check Cache
    is_cached = text_coverage.check_cache(selected_user, source_key, threshold=coverage_threshold)
    
    if is_cached and 'coverage_metrics' not in st.session_state:
        metrics, df_matches, _ = text_coverage.load_cache(selected_user, source_key, threshold=coverage_threshold)
        if metrics:
            st.session_state['coverage_metrics'] = metrics
            st.session_state['coverage_matches'] = df_matches
            st.success("Coverage results loaded from cache.")
            
    if st.button("Calculate Coverage") or (is_cached and 'coverage_metrics' not in st.session_state): # Auto-trigger if logic fails or just rely on state
         # Actually if it's cached we loaded it above. The button is for re-calc or initial calc.
         if not metrics and comparison_text:
            with st.spinner("Calculating semantic coverage..."):
                 metrics, df_matches, _ = text_coverage.calculate_text_coverage_metrics(selected_user, source_key, texts_full, comparison_text, threshold=coverage_threshold)
                 st.session_state['coverage_metrics'] = metrics
                 st.session_state['coverage_matches'] = df_matches

    if 'coverage_metrics' in st.session_state:
        res = st.session_state['coverage_metrics']
        df = st.session_state['coverage_matches']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{res['precision']:.2f}")
        c2.metric("Recall (Coverage)", f"{res['recall']:.2f}")
        c3.metric("F1 Score", f"{res['f1']:.2f}")
        
        st.write(f"**{res['tp_recall']}** / **{res['n_full']}** posts covered.")
        
        # Sensitivity Analysis (Line Chart)
        sens_data = text_coverage.sensitivity_analysis_text(texts_full, comparison_text)
        if sens_data:
            import plotly.graph_objects as go
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
            
            fig_sens.add_vline(x=coverage_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
            fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.error("Could not generate sensitivity data.")
        
        # Matches Table
        if df is not None and not df.empty:
            st.write("#### Matches")
            # Filter for covered items
            st.dataframe(df[df['covered'] == True], use_container_width=True)
            
            st.write("#### Missed (Posts NOT covered)")
            st.dataframe(df[df['covered'] == False], use_container_width=True)
        else:
             st.warning("No detail data available for matches.")

# --- Main App UI ---

import general_statistics

def render_dataset_statistics(df, api_key):
    general_statistics.render_dataset_statistics(df, api_key)

import gpt_evaluator

def render_gpt_evaluation(selected_user, api_key):
    # Subheader removed

    st.caption("Compare 'Narrative Base' vs 'Narrative Trajectory' using an LLM Judge.")
    
    # 1. Load Narratives
    # 1. Load Narratives from split cache
    base_data = report_base.load_base_report(selected_user)
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
    
    narrative_base = ""
    narrative_traj = ""
    
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    
    if traj_data_loaded:
        # Reconstruct full trajectory report (Summary + Phases)
        summary_text = traj_data_loaded.get('trajectory_summary', '')
        phases_text = ""
        if 'phases' in traj_data_loaded:
            for phase in traj_data_loaded['phases']:
                phases_text += f"\n\nPhase {phase.get('phase_num')}: {phase.get('start_date')} to {phase.get('end_date')} (Delta: {phase.get('delta', 0):.2f})\n"
                phases_text += phase.get('narrative', '')
        
        narrative_traj = f"{summary_text}\n{phases_text}"
            
    if not narrative_base or not narrative_traj:
        st.warning("Narratives not found. Please generate them in 'Narrative Trajectory' first.")
        return

    # 2. Run Evaluation
    # Check for cache and auto-load if available (bypassing API key)
    cache_path = os.path.join(gpt_evaluator.CACHE_DIR, f"eval_{selected_user}.json")
    if os.path.exists(cache_path) and 'eval_result' not in st.session_state:
        # Load from cache using dummy key (safe because cache exists)
        result_json, mapping = gpt_evaluator.evaluate_reports(selected_user, narrative_base, narrative_traj, api_key="cached")
        if result_json:
            st.session_state['eval_result'] = result_json
            st.session_state['eval_mapping'] = mapping
            st.success("Evaluation results loaded from cache.")

    if st.button("Run GPT Evaluation"):
        if not api_key:
            st.error("API Key required.")
        else:
            with st.spinner("Running blind evaluation... (This may take a minute)"):
                result_json, mapping = gpt_evaluator.evaluate_reports(selected_user, narrative_base, narrative_traj, api_key)
                st.session_state['eval_result'] = result_json
                st.session_state['eval_mapping'] = mapping
    
    # 3. Display Results
    if 'eval_result' in st.session_state and 'eval_mapping' in st.session_state:
        res = st.session_state['eval_result']
        mapping = st.session_state['eval_mapping']
        
        # Determine Identifier for Base and Traj
        # mapping = { "A": "base", "B": "trajectory" } etc
        id_base = "A" if mapping["A"] == "base" else "B"
        id_traj = "A" if mapping["A"] == "trajectory" else "B"
        
        # Winner
        preferred = res.get("Preferred_Report", "Tie")
        winner_label = "Tie"
        if preferred == id_base: winner_label = "Narrative Base"
        elif preferred == id_traj: winner_label = "Narrative Trajectory"
        
        st.success(f"🏆 Winner: **{winner_label}** (Report {preferred})")
        
        st.write(f"**Rationale:** {res.get('Rationale', '')}")
        
        # Table Construction
        criteria_list = [
            "Trajectory_Coverage", 
            "Temporal_Coherence", 
            "Change_Point_Sensitivity", 
            "Segment_Level_Specificity", 
            "Overall_Preference"
        ]
        
        table_data = []
        scores_A = res.get("Report_A", {})
        scores_B = res.get("Report_B", {})
        justifications = res.get("Criterion_Justifications", {})
        
        for crit in criteria_list:
            score_base = scores_A.get(crit, 0) if id_base == "A" else scores_B.get(crit, 0)
            score_traj = scores_A.get(crit, 0) if id_traj == "A" else scores_B.get(crit, 0)
            
            table_data.append({
                "Criterion": crit.replace("_", " "),
                "Score (Base)": score_base,
                "Score (Trajectory)": score_traj,
                "Justification": justifications.get(crit, "")
            })
            
        st.table(pd.DataFrame(table_data))

def main():
    st.title("🧠 Behavioral Analysis Dashboard")
    st.markdown("### Reddit User Risk Assessment & Segmentation")
    
    # -- Sidebar --
    st.sidebar.header("Data & Control")
    
    # API Key for Topics/Evaluations
    # Priority: 1. Manual User Input, 2. Streamlit Secrets (for Cloud Deployment)
    manual_api_key = st.sidebar.text_input("OpenAI API Key (for Topics)", type="password", help="Required for extracting topics or generating qualitative summaries.")
    
    # Use secrets if available as fallback (wrapped in try-except to avoid crash if no secrets file exists)
    try:
        if manual_api_key:
            api_key = manual_api_key
        elif "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = ""
    except:
        api_key = manual_api_key if manual_api_key else ""

    
    # Automatic File Discovery
    # Consolidated paths in 06 app (Relocated to app/ subfolder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    rel_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    
    uploaded_file = None
    
    df = None
    if uploaded_file:
        df = data.load_data(uploaded_file)
    elif os.path.exists(abs_path):
        df = data.load_data(abs_path)
    elif os.path.exists(rel_path):
        df = data.load_data(rel_path)
    else:
        st.sidebar.warning(f"Default file not found. Checked:\\n- `{abs_path}`\\n- `{rel_path}`\\nPlease upload manually.")
        st.stop()
        
    if df is None:
        st.error("Failed to load data.")
        st.stop()
        
    analysis_mode = st.sidebar.radio("Analysis Level", ["🌍 Global Dataset Stats", "👤 Single User Analysis"])
    st.sidebar.markdown("---")

    if analysis_mode == "🌍 Global Dataset Stats":
        render_dataset_statistics(df, api_key)
        
    else: # Single User Analysis
        # User Selection cached
        # User Selection cached
        users = list(data.get_subject_ids(df))
        
        # Determined default index for user 2714
        default_index = 0
        target_user = 2714
        
        # Handle int/str mismatch in list
        if target_user in users:
            default_index = users.index(target_user)
        elif str(target_user) in users:
            default_index = users.index(str(target_user))
            
        selected_user = st.sidebar.selectbox("Select User (Subject ID)", users, index=default_index)
            

    
        # -- Global Parameters (Moved from Dashboard) --
        st.sidebar.markdown("### ⚙️ Analysis Parameters")
        half_life = st.sidebar.slider("EMA Half-Life (Days)", 1, 60, 15, help="Controls the decay of the Risk Score.")
        k_segments = st.sidebar.slider("K-Segments", 1, 20, 10, help="Number of segments for trajectory.")
    
        # -- Filter User Data cached --
        user_data = data.get_user_data(df, selected_user)
        
        if user_data.empty:
            st.warning("No data for selected user.")
            st.stop()
    
        # -- Global Calculations --
        # Calculate Risk Score & Segments once for the user
        risk_series = ema.calculate_risk_score(user_data, half_life)
        if not risk_series.empty:
             segments = segment.segment_time_series(risk_series, k_segments)
        else:
             segments = []
            
        # -- Header Stats --
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("User ID", str(selected_user))
        col1.metric("Total Posts", len(user_data))
        
        start_date = user_data["Date"].min().strftime("%Y-%m-%d")
        end_date = user_data["Date"].max().strftime("%Y-%m-%d")
        col2.metric("First Post", start_date)
        col2.metric("Last Post", end_date)
        
        # Placeholders for Risk Metrics
        avg_score_metric = col3.empty()
        peak_score_metric = col4.empty()
        current_risk_metric = col5.empty()
        
        # Calculate metrics for header
        avg_score = risk_series.mean() if not risk_series.empty else 0
        peak_score = risk_series.max() if not risk_series.empty else 0
        current_score = risk_series.iloc[-1] if not risk_series.empty else 0
        
        avg_score_metric.metric("Avg Risk Score", f"{avg_score:.4f}")
        peak_score_metric.metric("Peak Risk Score", f"{peak_score:.4f}")
        current_risk_metric.metric("Current Risk", f"{current_score:.4f}", delta=f"{current_score - avg_score:.4f}")
        
        st.markdown("---")
    
        # -- Sidebar Navigation --
        # Removed View Mode as requested, merging Overview into main list
        
        analysis_section = st.sidebar.radio(
            "User Analysis Section", 
            [
                "👤 User Overview",
                "📊 Risk Dashboard", 
                "📖 Narrative Trajectory", 
                "⚖️ GPT Evaluation",
                "🧩 Topic Analysis", 
                "🧩 Topic Coverage",
                "📄 Text Coverage",
                "🔍 Topic Clustering"
            ]
        )
        
        st.subheader(f"{analysis_section}")
        st.caption(f"Selected User: {selected_user}")
    
        
        if analysis_section == "👤 User Overview":
            render_global_overview(user_data)
            
        elif analysis_section == "📊 Risk Dashboard":
            # Pass pre-calculated data
            render_risk_dashboard(user_data, risk_series, segments, half_life)
        
        elif analysis_section == "📖 Narrative Trajectory":
            # Pass pre-calculated segments
            render_trajectory_section(selected_user, user_data, segments, api_key)
    
        elif analysis_section == "⚖️ GPT Evaluation":
            render_gpt_evaluation(selected_user, api_key)
    
        elif analysis_section == "🧩 Topic Analysis":
            render_topic_analysis(selected_user, user_data, api_key)
    
        elif analysis_section == "🧩 Topic Coverage":
            render_topic_coverage(selected_user)
    
        elif analysis_section == "📄 Text Coverage":
            render_text_coverage(selected_user, user_data)
    
        elif analysis_section == "🔍 Topic Clustering":
            render_clustering_section(selected_user, user_data)
            


if __name__ == "__main__":
    main()
