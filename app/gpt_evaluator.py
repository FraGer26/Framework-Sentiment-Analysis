import os
import json
import random
import streamlit as st
from openai import OpenAI
import hashlib
import pandas as pd

# Cache directory for evaluations
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "evaluations")

def get_aggregate_stats():
    """
    Aggregates statistics from all cached evaluations.
    Returns a dict with DataFrame and summary counts.
    """
    if not os.path.exists(CACHE_DIR):
        return None
        
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("eval_") and f.endswith(".json")]
    
    if not files:
        return None
        
    # Stats accumulation
    stats = {}
    total_evals = 0
    preference_counts = {"Base": 0, "Trajectory": 0, "Tie": 0}
    
    for filename in files:
        path = os.path.join(CACHE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            result = data.get("result", {})
            mapping = data.get("mapping", {})
            
            if not result or not mapping:
                continue
                
            total_evals += 1
            
            # Identify Report IDs
            # mapping: { "A": "base", "B": "trajectory" }
            id_base = "A" if mapping.get("A") == "base" else "B"
            id_traj = "A" if mapping.get("A") == "trajectory" else "B"
            
            # Scores
            scores_base = result.get(f"Report_{id_base}", {})
            scores_traj = result.get(f"Report_{id_traj}", {})
            
            # Preference
            pref = result.get("Preferred_Report", "Tie")
            if pref == id_base:
                preference_counts["Base"] += 1
            elif pref == id_traj:
                preference_counts["Trajectory"] += 1
            else:
                preference_counts["Tie"] += 1
            
            # Iterate criteria
            for criteria in scores_base.keys():
                if criteria not in stats:
                    stats[criteria] = {"Base_Sum": 0, "Traj_Sum": 0, "Count": 0}
                
                stats[criteria]["Base_Sum"] += scores_base.get(criteria, 0)
                stats[criteria]["Traj_Sum"] += scores_traj.get(criteria, 0)
                stats[criteria]["Count"] += 1
                
        except Exception as e:
            # print(f"Error reading {filename}: {e}")
            continue
            
    if total_evals == 0:
        return None
        
    # Build DataFrame
    rows = []
    for crit, val in stats.items():
        count = val["Count"]
        if count > 0:
            avg_base = val["Base_Sum"] / count
            avg_traj = val["Traj_Sum"] / count
            
            rows.append({
                "Criterion": crit.replace("_", " "),
                "Avg Score (Base)": f"{avg_base:.2f}",
                "Avg Score (Trajectory)": f"{avg_traj:.2f}"
            })
            
    df = pd.DataFrame(rows)
    
    return {
        "df": df,
        "total_evals": total_evals,
        "preferences": preference_counts
    }

SUMMARY_CACHE_FILE = "eval_summary_cache.json"

def get_summary_cache_path():
    return os.path.join(CACHE_DIR, SUMMARY_CACHE_FILE)

def load_qualitative_summary():
    path = get_summary_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

def generate_qualitative_summary(api_key):
    """
    Generates a qualitative summary of all evaluations using LLM.
    Aggregates all JSONs in cache and asks for a comparative summary.
    """
    if not os.path.exists(CACHE_DIR):
        return None
        
    # Load all evaluation files
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("eval_") and f.endswith(".json")]
    all_judge_data = []
    
    for filename in files:
        if filename == SUMMARY_CACHE_FILE: continue
        path = os.path.join(CACHE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # We only need the result part usually, or the whole thing?
                # The notebook loads the whole json.
                all_judge_data.append(data)
        except:
            pass
            
    if not all_judge_data:
        return None
        
    # Build Prompt
    json_str = json.dumps(all_judge_data, indent=2)
    
    # Truncate if too long? 10 files might be fine, but 100 might hit context limits.
    # For now assume it fits or simple truncation.
    if len(json_str) > 100000: # Safe guard roughly
        json_str = json_str[:100000] + "... (truncated)"
        
    prompt = f"""
        Input:

        You are given a list of JSON objects representing evaluations: {json_str}

        Task:

        Instead of evaluating each report individually, aggregate all reports into a single comparison.
        Generate exactly one sentence of justification for each evaluation criterion that summarizes the relative strengths and weaknesses across all reports.
        Each justification must:
        - Compare the reports collectively
        - Reflect the relative scores qualitatively
        - Be concise, neutral, and technical
        Do not mention numeric values.
        Do not restate the criterion name.
        Do not add explanations beyond what is requested.

        Output:

        Return a single JSON object with the following structure:

        {{
            "Criterion_Justifications": {{
            "Trajectory_Coverage": "Aggregated comparative sentence across all reports",
            "Temporal_Coherence": "Aggregated comparative sentence across all reports",
            "Change_Point_Sensitivity": "Aggregated comparative sentence across all reports",
            "Segment_Level_Specificity": "Aggregated comparative sentence across all reports",
            "Overall_Preference": "Aggregated comparative sentence summarizing overall preference"
            }}
        }}
    """
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Used gpt-5.1 in notebook, using gpt-4o as standard high quality or fallback to user pref
            messages=[
                {"role": "system", "content": "You are an expert analyst of evaluation reports."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # Save to Cache
        cache_path = get_summary_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)
            
        return result_json
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None
def get_cache_path(user_id, report_base, report_traj):
    # User requested to name the file after the user_id

    filename = f"eval_{user_id}.json"
    return os.path.join(CACHE_DIR, filename)

def evaluate_reports(user_id, report_base, report_traj, api_key):
    """
    Evaluates two reports (Base vs Trajectory) using a blind A/B test with an LLM Judge.
    Returns the raw JSON response and the mapping used.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    cache_path = get_cache_path(user_id, report_base, report_traj)
    
    # Check Cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return cached_data["result"], cached_data["mapping"]
        except:
            pass # If cache read fails, proceed to re-run
    
    # Randomize A/B
    if random.random() < 0.5:
        report_A = report_base
        report_B = report_traj
        mapping = {
            "A": "base",
            "B": "trajectory"
        }
    else:
        report_A = report_traj
        report_B = report_base
        mapping = {
            "A": "trajectory",
            "B": "base"
        }
        
    user_prompt = f"""
Evaluate the following two reports according to the instructions below.

INSTRUCTIONS:
Carefully read Report A and Report B.
Evaluate each report independently using a 5-point Likert scale
(1 = Very poor, 5 = Excellent).

CRITERIA:
1. Trajectory Coverage
2. Temporal Coherence
3. Sensitivity to Change Points
4. Segment-Level Specificity
5. Overall Preference

CRITERIA DEFINITIONS:
1. Trajectory Coverage: how well the report captures the main phases of the user’s mental-health–related history, rather than focusing only on isolated posts or a single time period.
2. Temporal Coherence: the extent to which the report clearly describes changes over time (e.g., worsening, improvement, stability) and maintains a logically consistent temporal narrative.
3. Sensitivity to Change Points:  how clearly the report identifies and explains important turning points in the trajectory, such as transitions from high to lower severity or vice versa.
4. Segment-Level Specificity: the amount of concrete, phase-specific detail included in the report, such as references to typical posts, themes, or coping strategies characteristic of each phase.
5. Overall Preference: an overall judgment of which report provides the most useful and coherent description of the user’s depression-related trajectory.

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "Report_A": {{
    "Trajectory_Coverage": X,
    "Temporal_Coherence": X,
    "Change_Point_Sensitivity": X,
    "Segment_Level_Specificity": X,
    "Overall_Preference": X
  }},
  "Report_B": {{
    "Trajectory_Coverage": X,
    "Temporal_Coherence": X,
    "Change_Point_Sensitivity": X,
    "Segment_Level_Specificity": X,
    "Overall_Preference": X
  }},
  "Preferred_Report": "A | B | Tie",
  "Criterion_Justifications": {{
    "Trajectory_Coverage": "Concise justification comparing A and B (2 sentences).",
    "Temporal_Coherence": "Concise justification comparing A and B (2 sentences).",
    "Change_Point_Sensitivity": "Concise justification comparing A and B (2 sentences).",
    "Segment_Level_Specificity": "Concise justification comparing A and B (2 sentences).",
    "Overall_Preference": "Concise justification comparing A and B (2 sentences)."
  }},

  "Rationale": "Concise justification (2–4 sentences)."
}}

REPORT A:
<<<
{report_A}
>>>

REPORT B:
<<<
{report_B}
>>>
"""

    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.1", 
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert, impartial evaluator of mental-health trajectory reports. "
                        "Judge reports solely based on their content. "
                        "Do not infer their origin. Output only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format={ "type": "json_object" }
        )
        
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # Inject mapping into result for transparency if needed (but UI uses returned mapping)
        result_json["Report_Mapping"] = mapping
        
        # Save to Cache
        cache_data = {
            "result": result_json,
            "mapping": mapping
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)
        
        return result_json, mapping
        
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
        return None, None
