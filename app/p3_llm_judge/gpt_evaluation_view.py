import streamlit as st
from p2_narrative_report import report_base, report_trajectory
from p3_llm_judge import gpt_evaluator
import os
import pandas as pd

def render_gpt_evaluation(selected_user, api_key):

    st.caption("Compare 'Narrative Base' vs 'Narrative Trajectory' using an LLM Judge.")
    
    # 1. Carica Narrative
    # Carica Narrative da cache divisa
    base_data = report_base.load_base_report(selected_user)
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
    
    narrative_base = ""
    narrative_traj = ""
    
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    
    if traj_data_loaded:
        # Ricostruisci report traiettoria completo (Sommario + Fasi)
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

    # 2. Esegui Valutazione
    # Controlla cache e auto-carica se disponibile (bypassando chiave API)
    cache_path = os.path.join(gpt_evaluator.CACHE_DIR, f"eval_{selected_user}.json")
    if os.path.exists(cache_path) and 'eval_result' not in st.session_state:
        # Carica da cache usando chiave dummy (sicuro perché cache esiste)
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
    
    # 3. Mostra Risultati
    if 'eval_result' in st.session_state and 'eval_mapping' in st.session_state:
        res = st.session_state['eval_result']
        mapping = st.session_state['eval_mapping']
        
        # Determina Identificatore per Base e Traiettoria
        # mapping = { "A": "base", "B": "trajectory" } etc
        id_base = "A" if mapping["A"] == "base" else "B"
        id_traj = "A" if mapping["A"] == "trajectory" else "B"
        
        # Vincitore
        preferred = res.get("Preferred_Report", "Tie")
        winner_label = "Tie"
        if preferred == id_base: winner_label = "Narrative Base"
        elif preferred == id_traj: winner_label = "Narrative Trajectory"
        
        st.success(f"🏆 Winner: **{winner_label}** (Report {preferred})")
        
        st.write(f"**Rationale:** {res.get('Rationale', '')}")
        
        # Costruzione Tabella
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
