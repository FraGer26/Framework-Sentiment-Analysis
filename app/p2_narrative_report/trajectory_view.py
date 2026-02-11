# =============================================================================
# MODULO: trajectory_view.py
# DESCRIZIONE: Renderizza la sezione "Narrative Analysis" nell'interfaccia Streamlit.
#              Gestisce sia il caricamento dalla cache che la generazione
#              di nuovi report Base e Trajectory tramite API OpenAI.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import re  # Libreria per espressioni regolari (usata per conteggio parole)
from p0_global import data  # Modulo locale per costanti
from p2_narrative_report import report_base, report_trajectory  # Moduli per generazione report


def render_trajectory_section(selected_user, user_data, segments, api_key):
    """
    Renderizza la sezione completa dell'analisi narrativa.
    
    Se i report sono in cache, li carica e mostra direttamente.
    Se non sono in cache, mostra un pulsante per generarli tramite API.
    
    Mostra tre sotto-sezioni:
    1. Overall Narrative (Base): narrativa senza segmentazione
    2. Trajectory Summary: riepilogo integrato delle fasi
    3. Phase-by-Phase Evolution: dettaglio di ogni fase in expander
    
    Args:
        selected_user: ID dell'utente selezionato
        user_data (pd.DataFrame): DataFrame con i dati dell'utente
        segments (list): Lista dei segmenti dalla segmentazione
        api_key (str): Chiave API OpenAI
    """
    
    # Verifica se entrambi i report (base e trajectory) sono in cache
    base_data = report_base.load_base_report(selected_user)  # Carica report base dalla cache
    traj_data = report_trajectory.load_trajectory_report(selected_user)  # Carica report trajectory
    # Flag: True se entrambi i report esistono in cache
    is_traj_cached = (base_data is not None) and (traj_data is not None)
    
    # ==========================================================================
    # CASO 1: REPORT GIÀ IN CACHE
    # Carica e visualizza i report salvati precedentemente
    # ==========================================================================
    if is_traj_cached:
        st.success("Narrative analysis available in cache. Loading...")  # Messaggio di successo
        with st.spinner("Loading narrative from cache..."):
            # Ricarica entrambi i report (dalla cache, non chiama API)
            b_data, _ = report_base.generate_base_report(selected_user, user_data.copy(), api_key)
            t_data, _ = report_trajectory.generate_trajectory_report(selected_user, user_data.copy(), segments, api_key)
            # Unisce i due dizionari in uno solo usando l'operatore di spacchettamento
            combined = {**b_data, **t_data} if b_data and t_data else None
            
            if combined:
                # --- Sezione Report Base ---
                base_text = combined.get('base_analysis', 'N/A')  # Estrae la narrativa base
                # Conta le parole usando regex: trova tutte le sequenze di caratteri alfanumerici
                wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                
                st.markdown("### 📝 Overall Narrative (Base)")  # Titolo sezione
                st.write(base_text)  # Mostra il testo della narrativa
                st.caption(f"Word Count: {wc_base}")  # Mostra il conteggio parole
                
                # --- Sezione Metriche Report Trajectory ---
                traj_summary = combined.get('trajectory_summary', 'N/A')  # Riepilogo della traiettoria
                phases = combined.get('phases', [])  # Lista delle fasi
                
                # Conta le parole nel riepilogo
                wc_summary = len(re.findall(r"\b\w+\b", traj_summary)) if traj_summary and traj_summary != 'N/A' else 0
                # Conta le parole totali in tutte le narrative delle fasi
                wc_phases = sum([len(re.findall(r"\b\w+\b", p['narrative'])) for p in phases])
                # Somma totale: riepilogo + tutte le fasi
                wc_total_traj = wc_summary + wc_phases
                
                st.markdown("### 🔄 Trajectory Summary")  # Titolo sezione riepilogo
                st.write(traj_summary)  # Mostra il riepilogo
                st.caption(f"Total Trajectory Word Count (Summary + Phases): {wc_total_traj}")  # Conteggio totale
                
                # --- Sezione Evoluzione Fase per Fase ---
                st.markdown("### 📅 Phase-by-Phase Evolution")  # Titolo sezione fasi
                for phase in combined.get('phases', []):
                    # Ogni fase è mostrata in un expander cliccabile
                    # Il titolo include: numero fase, date, delta del rischio
                    with st.expander(f"Phase {phase['phase_num']}: {phase['start_date']} to {phase['end_date']} (Delta: {phase['delta']:.4f})"):
                        st.write(phase['narrative'])  # Mostra la narrativa della fase
    
    # ==========================================================================
    # CASO 2: REPORT NON IN CACHE
    # Mostra pulsante per generare i report tramite API OpenAI
    # ==========================================================================
    else:
        # Pulsante per avviare la generazione
        if st.button("Generate Narrative Analysis"):
            # Verifica che la chiave API sia stata fornita
            if not api_key:
                st.error("Please provide an OpenAI API Key in the sidebar to generate narrative.")
            else:
                with st.spinner("Analyzing trajectory..."):
                    # Verifica che ci siano segmenti disponibili
                    if not segments:
                        st.warning("No segments available to analyze.")
                    else:
                        # Genera entrambi i report chiamando le API GPT
                        b_data, _ = report_base.generate_base_report(selected_user, user_data.copy(), api_key)
                        t_data, _ = report_trajectory.generate_trajectory_report(selected_user, user_data.copy(), segments, api_key)
                        # Unisce i risultati
                        combined = {**b_data, **t_data} if b_data and t_data else None
                        
                        if combined:
                            st.info("Analysis generated from API and saved to cache.")
                            
                            # --- Report Base (stessa logica del caso cache) ---
                            base_text = combined.get('base_analysis', 'N/A')
                            wc_base = len(re.findall(r"\b\w+\b", base_text)) if base_text and base_text != 'N/A' else 0
                            
                            st.markdown("### 📝 Overall Narrative (Base)")
                            st.write(base_text)
                            st.caption(f"Word Count: {wc_base}")
                            
                            # --- Metriche Report Trajectory ---
                            traj_summary = combined.get('trajectory_summary', 'N/A')
                            phases = combined.get('phases', [])
                            
                            wc_summary = len(re.findall(r"\b\w+\b", traj_summary)) if traj_summary and traj_summary != 'N/A' else 0
                            wc_phases = sum([len(re.findall(r"\b\w+\b", p['narrative'])) for p in phases])
                            wc_total_traj = wc_summary + wc_phases
                            
                            st.markdown("### 🔄 Trajectory Summary")
                            st.info(traj_summary)  # Usa st.info per evidenziare il riepilogo
                            st.caption(f"Total Trajectory Word Count (Summary + Phases): {wc_total_traj}")
                            
                            # --- Evoluzione Fase per Fase ---
                            st.markdown("### 📅 Phase-by-Phase Evolution")
                            for phase in combined.get('phases', []):
                                with st.expander(f"Phase {phase['phase_num']}: {phase['start_date']} to {phase['end_date']} (Delta: {phase['delta']:.4f})"):
                                    st.write(phase['narrative'])
                        else:
                            # Messaggio di errore se la generazione fallisce
                            st.error("Failed to generate analysis.")
