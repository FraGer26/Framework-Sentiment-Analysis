# Relazione del Progetto: Framework Sentiment Analysis

## Informazioni Generali

| Campo | Valore |
|-------|--------|
| **Titolo** | Framework Sentiment Analysis Dashboard |
| **Tecnologia** | Python, Streamlit, PySpark |
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
│   │ A/B Testing │  │ GPT Extraction  │  │ Semantic Matching  │              │
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
- Calcolo statistiche globali dataset tramite Apache Spark SQL
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
smoothed = rolling_mean(score, window=7, center=True, min_periods=1)
```

**Segmentazione Top-Down Piecewise Linear**:
- Algoritmo ricorsivo che divide la serie temporale
- Trova il punto di massima deviazione dalla linea retta (distanza perpendicolare punto-retta)
- Divide in K segmenti ottimali (default K=10)

---

### 3.3 Modulo p2_narrative_report - Report Narrativi

**File**: `report_base.py`, `report_trajectory.py`, `trajectory_view.py`

**Tipi di Report**:

| Report | Input | Descrizione |
|--------|-------|-------------|
| **Base** | Post completi + Timestamp | Analisi globale evoluzione emotiva senza segmentazione |
| **Trajectory** | Segmenti + Post per fase + Breakpoint EMA | Analisi per fase temporale con traiettoria |

**Prompt GPT per Report Base**:
```
Task: Analyze posts as time series describing user's emotional state
- Identify depressive-leaning expressions
- Describe emotional tone changes from beginning to end
- Create integrative narrative about emotional evolution
Output: Single analytical narrative, no bullet points
```

**Prompt GPT per Report Trajectory**:
- Genera un'analisi narrativa per ciascuna fase temporale (segmento)
- Produce un riepilogo integrato della traiettoria complessiva

---

### 3.4 Modulo p3_llm_judge - Valutazione A/B

**File**: `gpt_evaluator.py`, `gpt_evaluation_view.py`

**Metodologia**:
1. Test A/B cieco tra Report Base e Report Trajectory
2. Randomizzazione ordine presentazione
3. Valutazione LLM multi-criterio su scala Likert 1-5 (1 = Very poor, 5 = Excellent)
4. Temperatura impostata a 0.0 per massima riproducibilità

**Criteri di Valutazione (5)**:

| Criterio | Definizione |
|----------|-------------|
| **Trajectory Coverage** | Quanto il report cattura le fasi principali della storia dell'utente |
| **Temporal Coherence** | Chiarezza nel descrivere i cambiamenti nel tempo (peggioramento, miglioramento, stabilità) |
| **Change Point Sensitivity** | Capacità di identificare e spiegare i punti di svolta nella traiettoria |
| **Segment-Level Specificity** | Dettaglio concreto per ogni fase (riferimenti a post, temi, strategie di coping) |
| **Overall Preference** | Giudizio complessivo su quale report sia più utile e coerente |

**Output**: JSON con punteggi per criterio, preferenza (A/B/Tie), giustificazioni per ogni criterio e razionale complessivo

---

### 3.5 Modulo p4_topic_analysis - Estrazione Argomenti

**File**: `topic_model.py`, `topic_analysis_view.py`

**Metodo**: Estrazione tramite GPT (non BERTopic). L'utente può scegliere la sorgente del testo: post grezzi, narrativa base, o narrativa trajectory.

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
| **Precision** | TP_sample / N_sample | % topics candidati che matchano con un riferimento |
| **Recall** | TP_full / N_full | % topics di riferimento coperti da almeno un candidato |
| **F1** | 2×P×R / (P+R) | Media armonica |

**Matching Semantico**: Similarità coseno con threshold configurabile (default 0.75)

**Modello Embedding**: `nomic-ai/modernbert-embed-base`

---

### 3.7 Modulo p6_text_coverage - Copertura Testuale

**File**: `text_coverage.py`, `embedding_utils.py`, `text_coverage_view.py`

**Algoritmo**:
1. Embedding dei post originali tramite SentenceTransformer
2. Embedding del report narrativo
3. Matrice similarità coseno (n_post × n_componenti_narrativa)
4. Match con threshold (default 0.5)

**Embedding**: `nomic-ai/modernbert-embed-base`, normalizzato, batch_size=16

**Analisi di Sensibilità**: Curva F1/Precision/Recall vs Threshold (0.0 → 1.0, step 0.05)

---

### 3.8 Modulo p7_topic_analysis_clustering - Clustering

**File**: `clustering.py`, `clustering_view.py`

**Pipeline Manuale (senza BERTopic)**:
```
Post → SentenceTransformer → UMAP (5D) → HDBSCAN → c-TF-IDF → Cluster Assignment
                ↓
        all-mpnet-base-v2
```

**Parametri UMAP (Clustering)**:
- n_neighbors: 40
- n_components: 5
- min_dist: 0.0
- metric: cosine
- random_state: 42

**Parametri UMAP (Visualizzazione 2D)**:
- n_neighbors: 15
- n_components: 2
- min_dist: 0.0
- metric: cosine

**Parametri HDBSCAN**:
- min_cluster_size: 35
- metric: euclidean
- cluster_selection_method: eom (Excess of Mass)

**Keyword Extraction**: c-TF-IDF (class-based TF-IDF) con CountVectorizer (stop_words="english"), top 10 parole per cluster

**Mappatura Ground Truth**: I cluster vengono mappati ai topic ground truth (estratti nella fase p4) tramite similarità coseno degli embedding

**Visualizzazione**: Plotly scatter 2D con proiezione UMAP, annotazioni centroidi e filtro outlier (Topic -1)

---

## 4. Stack Tecnologico

| Categoria | Tecnologie |
|-----------|------------|
| **Frontend** | Streamlit 1.31.1 |
| **Data Processing** | Pandas 2.2.0, NumPy 1.26.4, PySpark (Apache Spark SQL) |
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
│   ├── data/             # Dati utente filtrati (CSV)
│   ├── ema/              # Score EMA calcolati (JSON)
│   ├── segments/         # Segmenti temporali (JSON)
│   └── global/           # Statistiche globali Spark (JSON)
├── clusters/             # Risultati UMAP + HDBSCAN (JSON)
├── coverage/             # Metriche copertura testo/topic (JSON)
├── evaluations/          # Valutazioni A/B LLM (JSON)
├── reports/              # Report narrativi generati
│   ├── base/             # Report base (JSON)
│   └── trajectory/       # Report traiettoria (JSON)
└── topics/               # Argomenti estratti GPT (JSON)
```

---

## 6. Funzionalità Dashboard

### Modalità Analisi

1. **👤 Analisi Singolo Utente**
   - Selezione utente per Subject ID (default: 2714)
   - Parametri configurabili (EMA half-life 1-60 giorni, K-segmenti 1-20)
   - 8 sezioni di analisi dettagliata


2. **🌍 Statistiche Globali Dataset**
   - Panoramica aggregata calcolata con Spark SQL
   - Distribuzione sentiment nel tempo
   - Rankings utenti per attività e rischio
   - Valutazione qualitativa GPT aggregata

### Sezioni Utente

| Sezione | Contenuto |
|---------|-----------|
| 👤 Panoramica | Timeline attività, ultimi 10 post, top 5 post a rischio |
| 📊 Dashboard Rischio | Grafico EMA colorato per segmento, tabella breakpoint |
| 📖 Traiettoria Narrativa | Report Base vs Trajectory generati con GPT |
| ⚖️ Valutazione GPT | Risultati test A/B cieco su 5 criteri |
| 🧩 Analisi Argomenti | Topic extraction granulare (positivi/neutri/negativi) |
| 🧩 Copertura Argomenti | Metriche P/R/F1 tematiche con analisi sensibilità |
| 📄 Copertura Testo | Analisi copertura semantica embedding |





---

## 7. Installazione e Uso

### Prerequisiti
- Python 3.9+
- Java JDK 11+ (per Apache Spark)
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

- **Caching Multi-Livello**: Streamlit `@st.cache_data` / `@st.cache_resource` + cache su disco JSON
- **Calcoli On-Demand**: Analisi costose (LLM, clustering, embedding) eseguite solo su richiesta utente
- **Batch Processing Embedding**: SentenceTransformer con `batch_size=16` e `normalize_embeddings=True`
- **Spark SQL**: Usato per statistiche globali aggregate (query distribuite su dataset completo)

### Limitazioni

- API OpenAI richiede connessione internet e crediti
- Dataset grande gestito con Git LFS
- Modelli embedding richiedono ~500MB download iniziale
- Apache Spark richiede Java JDK 11+ installato (`JAVA_HOME` configurato)

---

## 9. Conclusioni

Il Framework Sentiment Analysis rappresenta un sistema completo per l'analisi comportamentale multi-dimensionale, integrando:

- **Tecniche Tradizionali**: EMA, segmentazione piecewise linear top-down
- **Deep Learning**: Embedding SentenceTransformer, clustering UMAP + HDBSCAN, c-TF-IDF
- **LLM Generativi**: Report narrativi, estrazione topic granulare e valutazione qualitativa A/B cieca
- **Big Data**: Statistiche globali calcolate con Apache Spark SQL

L'architettura modulare (8 fasi in pipeline) permette estensibilità e manutenibilità, mentre il sistema di caching multi-livello (Streamlit + disco) garantisce performance ottimali anche con dataset estesi.

---

**Autore**: Franco  
**Data**: Febbraio 2026  
**Versione**: 1.1
