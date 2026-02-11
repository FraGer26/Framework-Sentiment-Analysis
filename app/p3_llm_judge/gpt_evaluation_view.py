# =============================================================================
# MODULO: gpt_evaluation_view.py
# DESCRIZIONE: Renderizza la sezione di valutazione GPT nell'interfaccia Streamlit.
#              Permette di eseguire un test cieco A/B tra il report Base e
#              il report Trajectory, mostrando punteggi, vincitore e giustificazioni.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import os  # Libreria per operazioni su file e percorsi
import pandas as pd  # Libreria per manipolazione dati tabulari
from p2_narrative_report import report_base, report_trajectory  # Moduli per caricare i report
from p3_llm_judge import gpt_evaluator  # Modulo per la valutazione GPT


def render_gpt_evaluation(selected_user, api_key):
    """
    Renderizza la sezione completa della valutazione GPT.
    
    Mostra:
    1. Il vincitore della valutazione (Base vs Trajectory vs Tie)
    2. La motivazione della preferenza
    3. Una tabella con i punteggi dettagliati per ogni criterio
    
    Se la valutazione è in cache, la carica automaticamente.
    Altrimenti, mostra un pulsante per eseguirla.
    
    Args:
        selected_user: ID dell'utente selezionato
        api_key (str): Chiave API OpenAI
    """

    # Descrizione della sezione
    st.caption("Compare 'Narrative Base' vs 'Narrative Trajectory' using an LLM Judge.")
    
    # ==========================================================================
    # FASE 1: CARICAMENTO NARRATIVE DALLA CACHE
    # Carica entrambi i report per preparare l'input alla valutazione
    # ==========================================================================
    base_data = report_base.load_base_report(selected_user)  # Carica report base
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)  # Carica report trajectory
    
    # Inizializza le stringhe delle narrative
    narrative_base = ""  # Testo del report base
    narrative_traj = ""  # Testo del report trajectory
    
    # Estrae la narrativa base dal dizionario caricato
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    
    # Ricostruisce il testo completo del report trajectory
    # (riepilogo + tutte le narrative delle fasi)
    if traj_data_loaded:
        # Estrae il riepilogo generale
        summary_text = traj_data_loaded.get('trajectory_summary', '')
        phases_text = ""
        # Ricostruisce il testo delle fasi
        if 'phases' in traj_data_loaded:
            for phase in traj_data_loaded['phases']:
                # Aggiunge l'intestazione della fase con numero, date e delta
                phases_text += f"\n\nPhase {phase.get('phase_num')}: {phase.get('start_date')} to {phase.get('end_date')} (Delta: {phase.get('delta', 0):.2f})\n"
                # Aggiunge la narrativa della fase
                phases_text += phase.get('narrative', '')
        
        # Combina riepilogo e fasi in un unico testo
        narrative_traj = f"{summary_text}\n{phases_text}"
    
    # Se una delle narrative manca, mostra un avviso e interrompe
    if not narrative_base or not narrative_traj:
        st.warning("Narratives not found. Please generate them in 'Narrative Trajectory' first.")
        return

    # ==========================================================================
    # FASE 2: CARICAMENTO AUTOMATICO DALLA CACHE
    # Se la valutazione esiste in cache, la carica nello stato della sessione
    # ==========================================================================
    cache_path = os.path.join(gpt_evaluator.CACHE_DIR, f"eval_{selected_user}.json")
    # Carica dalla cache solo se il file esiste e non è già nello stato sessione
    if os.path.exists(cache_path) and 'eval_result' not in st.session_state:
        # Chiama evaluate_reports con chiave dummy "cached" perché la cache verrà usata
        result_json, mapping = gpt_evaluator.evaluate_reports(selected_user, narrative_base, narrative_traj, api_key="cached")
        if result_json:
            # Salva nello stato sessione di Streamlit per persistenza tra refresh
            st.session_state['eval_result'] = result_json
            st.session_state['eval_mapping'] = mapping
            st.success("Evaluation results loaded from cache.")

    # ==========================================================================
    # FASE 2B: PULSANTE PER NUOVA VALUTAZIONE
    # Permette di eseguire/rieseguire la valutazione tramite API
    # ==========================================================================
    if st.button("Run GPT Evaluation"):
        if not api_key:
            st.error("API Key required.")
        else:
            with st.spinner("Running blind evaluation... (This may take a minute)"):
                # Esegue la valutazione cieca A/B
                result_json, mapping = gpt_evaluator.evaluate_reports(selected_user, narrative_base, narrative_traj, api_key)
                # Salva i risultati nello stato sessione
                st.session_state['eval_result'] = result_json
                st.session_state['eval_mapping'] = mapping
    
    # ==========================================================================
    # FASE 3: VISUALIZZAZIONE RISULTATI
    # Mostra vincitore, motivazione e tabella punteggi dettagliati
    # ==========================================================================
    if 'eval_result' in st.session_state and 'eval_mapping' in st.session_state:
        # Recupera risultati e mappatura dallo stato sessione
        res = st.session_state['eval_result']
        mapping = st.session_state['eval_mapping']
        
        # Determina quale ID (A o B) corrisponde a Base e Trajectory
        id_base = "A" if mapping["A"] == "base" else "B"  # ID del report base
        id_traj = "A" if mapping["A"] == "trajectory" else "B"  # ID del report trajectory
        
        # Determina il vincitore traducendo l'ID in un'etichetta leggibile
        preferred = res.get("Preferred_Report", "Tie")  # ID del vincitore (A, B, o Tie)
        winner_label = "Tie"  # Etichetta predefinita: pareggio
        if preferred == id_base: winner_label = "Narrative Base"  # Se vince il base
        elif preferred == id_traj: winner_label = "Narrative Trajectory"  # Se vince il trajectory
        
        # Mostra il vincitore con enfasi
        st.success(f"🏆 Winner: **{winner_label}** (Report {preferred})")
        
        # Mostra la motivazione della preferenza
        st.write(f"**Rationale:** {res.get('Rationale', '')}")
        
        # =================================================================
        # COSTRUZIONE TABELLA PUNTEGGI
        # Crea una tabella con i punteggi per ogni criterio
        # =================================================================
        # Lista dei 5 criteri di valutazione
        criteria_list = [
            "Trajectory_Coverage",  # Copertura della traiettoria
            "Temporal_Coherence",  # Coerenza temporale
            "Change_Point_Sensitivity",  # Sensibilità ai punti di svolta
            "Segment_Level_Specificity",  # Specificità per segmento
            "Overall_Preference"  # Preferenza complessiva
        ]
        
        table_data = []  # Lista per costruire le righe della tabella
        scores_A = res.get("Report_A", {})  # Punteggi del report A
        scores_B = res.get("Report_B", {})  # Punteggi del report B
        justifications = res.get("Criterion_Justifications", {})  # Giustificazioni per criterio
        
        # Per ogni criterio, estrae i punteggi corretti (base e trajectory)
        for crit in criteria_list:
            # Associa i punteggi di A/B ai report corretti usando la mappatura
            score_base = scores_A.get(crit, 0) if id_base == "A" else scores_B.get(crit, 0)
            score_traj = scores_A.get(crit, 0) if id_traj == "A" else scores_B.get(crit, 0)
            
            # Aggiunge la riga alla tabella
            table_data.append({
                "Criterion": crit.replace("_", " "),  # Nome leggibile del criterio
                "Score (Base)": score_base,  # Punteggio del report base
                "Score (Trajectory)": score_traj,  # Punteggio del report trajectory
                "Justification": justifications.get(crit, "")  # Giustificazione del giudice
            })
        
        # Mostra la tabella dei punteggi nell'interfaccia
        st.table(pd.DataFrame(table_data))
