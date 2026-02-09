# Relazione del Progetto: Framework Sentiment Analysis

## Informazioni Generali

| Campo | Valore |
|-------|--------|
| **Titolo** | Framework Sentiment Analysis Dashboard |
| **Tecnologia** | Python, Streamlit |
| **Dominio** | Analisi Comportamentale e Sentiment Analysis su dati Reddit |
| **Modello LLM** | OpenAI GPT-5.1 |

---

## 1. Introduzione e Obiettivi

Questo progetto implementa una **dashboard analitica multi-modale** per la valutazione e visualizzazione del comportamento utente, delle traiettorie emotive e degli interessi tematici su Reddit.

### Obiettivi Principali

1. **Classificazione del Sentiment**: Analisi automatica dei post per identificare stati emotivi (depressione severa, moderata, neutro)
2. **Segmentazione Temporale**: Divisione della timeline utente in segmenti comportamentali distinti
3. **Generazione Report Narrativi**: Creazione di riassunti qualitativi usando LLM
4. **Topic Modeling**: Estrazione di argomenti granulari dai contenuti testuali
5. **Metriche di Copertura**: Valutazione della qualità dei report generati

---

## 2. Architettura del Sistema

Il sistema è organizzato in **8 moduli funzionali** (fasi), seguendo un pattern di pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT APP                                   │
│                               (app.py)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐  ┌─────────────────┐  ┌────────────────────┐              │
│   │  p0_global  │  │ p1_segmentation │  │ p2_narrative_report│              │
│   │ Data Layer  │──│ EMA + Segments  │──│ GPT Reports        │              │
│   └─────────────┘  └─────────────────┘  └────────────────────┘              │
│          │                  │                     │                          │
│          ▼                  ▼                     ▼                          │
│   ┌─────────────┐  ┌─────────────────┐  ┌────────────────────┐              │
│   │ p3_llm_judge│  │ p4_topic_analysis│  │ p5_topic_coverage  │              │
│   │ A/B Testing │  │ BERTopic + GPT  │  │ Semantic Matching  │              │
│   └─────────────┘  └─────────────────┘  └────────────────────┘              │
│          │                  │                     │                          │
│          ▼                  ▼                     ▼                          │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │      p6_text_coverage      │    p7_topic_clustering       │             │
│   │      Embedding Coverage    │    UMAP + HDBSCAN            │             │
│   └────────────────────────────┴──────────────────────────────┘             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MULTI-LEVEL CACHE                                  │
│        calculation/ │ clusters/ │ coverage/ │ evaluations/ │ reports/       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Descrizione dei Moduli

### 3.1 Modulo p0_global - Data Layer

**File**: `data.py`, `queries.py`, `general_statistics.py`, `overview_view.py`, `dataset_stats_view.py`

**Funzionalità**:
- Caricamento dati classificati (`output_Classification.csv`)
- Caching multi-livello su disco (JSON/CSV)
- Calcolo statistiche globali dataset
- Aggregazione rischio giornaliero: `risk = 2 × P(Severe) + P(Moderate)`

**Formula Rischio**:
```
raw_risk = 2 × Prob_Severe_Depressed + Prob_Moderate_Depressed
daily_risk = mean(raw_risk) per giorno
```

---

### 3.2 Modulo p1_segmentation - Analisi Temporale

**File**: `ema.py`, `segment.py`, `risk_view.py`

**Algoritmo EMA (Exponential Moving Average)**:
```
α = 0.5^(1/half_life)
score_t = α × score_{t-1} + (1-α) × value_t
smoothed = rolling_mean(score, window=7, center=True)
```

**Segmentazione Top-Down Piecewise Linear**:
- Algoritmo ricorsivo che divide la serie temporale
- Trova il punto di massima deviazione dalla linea retta
- Divide in K segmenti ottimali

---

### 3.3 Modulo p2_narrative_report - Report Narrativi

**File**: `report_base.py`, `report_trajectory.py`, `trajectory_view.py`

**Tipi di Report**:

| Report | Input | Descrizione |
|--------|-------|-------------|
| **Base** | Post completi + Timestamp | Analisi globale evoluzione emotiva |
| **Trajectory** | Segmenti + Post per segmento | Analisi per fase temporale |

**Prompt GPT per Report Base**:
```
Task: Analyze posts as time series describing user's emotional state
- Identify depressive-leaning expressions
- Describe emotional tone changes from beginning to end
- Create integrative narrative about emotional evolution
Output: Single analytical narrative, no bullet points
```

---

### 3.4 Modulo p3_llm_judge - Valutazione A/B

**File**: `gpt_evaluator.py`, `gpt_evaluation_view.py`

**Metodologia**:
1. Test A/B cieco tra Report Base e Report Trajectory
2. Randomizzazione ordine presentazione
3. Valutazione LLM multi-criterio:
   - Coherence (1-5)
   - Completeness (1-5)
   - Emotional Accuracy (1-5)
   - Overall Preference

**Output**: JSON con punteggi e giustificazioni

---

### 3.5 Modulo p4_topic_analysis - Estrazione Argomenti

**File**: `topic_model.py`, `topic_analysis_view.py`

**Prompt Granular Topic Extraction**:
```
Extract specific, concrete, distinct topics
- No generalization, preserve context
- Phrases of 6-15 words
- 10-20 distinct topics
- Classify: positive / neutral / negative

Output: {"positivetopics": [...], "neutraltopics": [...], "negativetopics": [...]}
```

---

### 3.6 Modulo p5_topic_coverage - Copertura Tematica

**File**: `topic_coverage.py`, `topic_coverage_view.py`

**Metriche di Copertura**:

| Metrica | Formula | Significato |
|---------|---------|-------------|
| **Precision** | TP_sample / N_sample | % topics campione che matchano |
| **Recall** | TP_full / N_full | % topics pieni coperti |
| **F1** | 2×P×R / (P+R) | Media armonica |

**Matching Semantico**: Similarità coseno con threshold configurabile (default 0.75)

**Modello Embedding**: `nomic-ai/modernbert-embed-base`

---

### 3.7 Modulo p6_text_coverage - Copertura Testuale

**File**: `text_coverage.py`, `embedding_utils.py`, `text_coverage_view.py`

**Algoritmo**:
1. Embedding dei post originali
2. Embedding del report narrativo
3. Matrice similarità coseno
4. Match con threshold (default 0.5)

**Analisi di Sensibilità**: Curva F1 vs Threshold (0.0 → 1.0)

---

### 3.8 Modulo p7_topic_analysis_clustering - Clustering

**File**: `clustering.py`, `clustering_view.py`

**Pipeline Manuale**:
```
Post → SentenceTransformer → UMAP → HDBSCAN → Cluster Assignment
                    ↓
            all-mpnet-base-v2
```

**Parametri UMAP**:
- n_neighbors: 15
- n_components: 5
- min_dist: 0.0
- metric: cosine

**Visualizzazione**: Plotly scatter 2D con proiezione UMAP

---

## 4. Stack Tecnologico

| Categoria | Tecnologie |
|-----------|------------|
| **Frontend** | Streamlit 1.31.1 |
| **Data Processing** | Pandas 2.2.0, NumPy 1.26.4 |
| **Machine Learning** | Scikit-learn 1.4.0, UMAP 0.5.5, HDBSCAN 0.8.33 |
| **NLP** | Sentence-Transformers 2.3.1, BERTopic 0.16.0 |
| **LLM** | OpenAI API 1.12.0 (GPT-5.1) |
| **Visualizzazione** | Plotly 5.18.0, Matplotlib 3.8.2, Seaborn 0.13.2 |

---

## 5. Struttura Dati

### Dataset Input

**File**: `classification/output_Classification.csv`

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| Subject ID | int | Identificativo utente |
| Chunk | int | Indice del chunk testuale |
| Date | datetime | Data del post |
| Text | str | Contenuto testuale |
| Prob_Severe_Depressed | float | Probabilità depressione severa |
| Prob_Moderate_Depressed | float | Probabilità depressione moderata |

### Sistema di Cache

```
cache/
├── calculation/          # Preprocessamento numerico
│   ├── data/             # Dati utente filtrati
│   ├── ema/              # Score EMA calcolati
│   ├── segments/         # Segmenti temporali
│   └── global/           # Statistiche globali
├── clusters/             # Risultati BERTopic/HDBSCAN
├── coverage/             # Metriche copertura testo/topic
├── evaluations/          # Valutazioni A/B LLM
├── reports/              # Report narrativi generati
│   ├── base/             # Report base
│   └── trajectory/       # Report traiettoria
└── topics/               # Argomenti estratti GPT
```

---

## 6. Funzionalità Dashboard

### Modalità Analisi

1. **👤 Analisi Singolo Utente**
   - Selezione utente per Subject ID
   - Parametri configurabili (EMA half-life, K-segmenti)
   - 8 sezioni di analisi dettagliata

2. **🌍 Statistiche Globali Dataset**
   - Panoramica aggregata
   - Distribuzione sentiment nel tempo

### Sezioni Utente

| Sezione | Contenuto |
|---------|-----------|
| 👤 Panoramica | Statistiche base, timeline post |
| 📊 Dashboard Rischio | Grafico EMA, visualizzazione segmenti |
| 📖 Traiettoria Narrativa | Report Base vs Trajectory |
| ⚖️ Valutazione GPT | Risultati test A/B cieco |
| 🧩 Analisi Argomenti | Topic extraction granulare |
| 🧩 Copertura Argomenti | Metriche P/R/F1 tematiche |
| 📄 Copertura Testo | Analisi copertura embedding |
| 🔍 Clustering | Visualizzazione UMAP clusters |

---

## 7. Installazione e Uso

### Prerequisiti
- Python 3.9+
- Git LFS (per dataset)
- OpenAI API Key

### Installazione
```bash
git clone <repository-url>
cd "Framework Sentiment Analysis"
pip install -r requirements.txt
git lfs pull
```

### Esecuzione
```bash
streamlit run app/app.py
```

### Deployment Streamlit Cloud
1. Push repository su GitHub
2. Connettere a Streamlit Cloud
3. Configurare `app/app.py` come main file
4. Aggiungere OpenAI API Key nei Secrets

---

## 8. Considerazioni Tecniche

### Ottimizzazioni

- **Caching Multi-Livello**: Streamlit `@st.cache_data` + cache su disco JSON
- **Calcoli On-Demand**: Analisi costose (LLM, clustering) eseguite solo su richiesta utente
- **Batch Processing Embedding**: Sentence-transformers gestisce batch automaticamente

### Limitazioni

- API OpenAI richiede connessione internet e crediti
- Dataset grande gestito con Git LFS
- Modelli embedding richiedono ~500MB download iniziale

---

## 9. Conclusioni

Il Framework Sentiment Analysis rappresenta un sistema completo per l'analisi comportamentale multi-dimensionale, integrando:

- **Tecniche Tradizionali**: EMA, segmentazione piecewise linear
- **Deep Learning**: Embedding BERT, clustering HDBSCAN
- **LLM Generativi**: Report narrativi e valutazione qualitativa

L'architettura modulare permette estensibilità e manutenibilità, mentre il sistema di caching garantisce performance ottimali anche con dataset estesi.

---

**Autore**: Franco  
**Data**: Febbraio 2026  
**Versione**: 1.0
