import streamlit as st
import report_base
import report_trajectory
import re

def render_trajectory_section(selected_user, user_data, segments, api_key):
    
    base_data = report_base.load_base_report(selected_user)
    traj_data = report_trajectory.load_trajectory_report(selected_user)
    is_traj_cached = (base_data is not None) and (traj_data is not None)
    
    if is_traj_cached:
        st.success("Narrative analysis available in cache. Loading...")
        with st.spinner("Loading narrative from cache..."):
            # Carica entrambe le parti
            b_data, _ = report_base.generate_base_report(selected_user, user_data.copy(), api_key)
            t_data, _ = report_trajectory.generate_trajectory_report(selected_user, user_data.copy(), segments, api_key)
            combined = {**b_data, **t_data} if b_data and t_data else None
            
            if combined:
                 # Report Base
                 base_text = combined.get('base_analysis', 'N/A')
                 wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                 
                 st.markdown("### 📝 Overall Narrative (Base)")
                 st.write(base_text)
                 st.caption(f"Word Count: {wc_base}")
                 
                 # Metriche Report Traiettoria
                 traj_summary = combined.get('trajectory_summary', 'N/A')
                 phases = combined.get('phases', [])
                 
                 wc_summary = len(re.findall(r"\b\w+\b", traj_summary)) if traj_summary and traj_summary != 'N/A' else 0
                 wc_phases = sum([len(re.findall(r"\b\w+\b", p['narrative'])) for p in phases])
                 wc_total_traj = wc_summary + wc_phases
                 
                 st.markdown("### 🔄 Trajectory Summary")
                 # Modificato da info a write per coerenza
                 st.write(traj_summary) 
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
                            
                            # Report Base
                            base_text = combined.get('base_analysis', 'N/A')
                            wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                            
                            st.markdown("### 📝 Overall Narrative (Base)")
                            st.write(base_text)
                            st.caption(f"Word Count: {wc_base}")
                            
                            # Metriche Report Traiettoria
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
