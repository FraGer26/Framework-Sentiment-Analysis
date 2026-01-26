# Reddit User Analytics Dashboard

A sophisticated multi-modal analytics tool designed to evaluate and visualize user behavior, emotional trajectories, and topical interests on Reddit.

## 🚀 Overview

This application provides a comprehensive suite of tools for researchers and data analysts to dive deep into Reddit user data. It leverages Large Language Models (LLMs) and advanced natural language processing (NLP) techniques to generate qualitative insights, perform clustering, and track user sentiment over time.

### Key Features

- **Qualitative GPT Summarization**: Generates high-level summaries of user evaluations.
- **Dynamic Topic Modeling**: Uses BERTopic and GPT to extract granular interests.
- **Trajectory Analysis**: Visualizes emotional and behavioral shifts across time segments.
- **Coverage Metrics**: Evaluates the linguistic and topical breadth of user interactions.
- **Global Statistics Dashboard**: Provides a bird's-eye view of dataset metrics.

## 📁 Project Structure

```text
06 app/
├── app/                    # Source code directory
│   ├── app.py              # Main Streamlit application entry point
│   ├── clustering.py       # BERTopic clustering logic
│   ├── data.py             # Data loading and preprocessing utilities
│   ├── ema.py              # Exponential Moving Average calculations
│   ├── embedding_utils.py  # Utilities for handling text embeddings
│   ├── gpt_evaluator.py    # LLM-based evaluation and summarization
│   ├── report_base.py      # Base report generation logic
│   ├── report_trajectory.py # Trajectory-based report generation
│   ├── segment.py          # Time-series segmentation algorithms
│   ├── text_coverage.py    # Textual density and coverage metrics
│   ├── topic_coverage.py   # Topical diversity metrics
│   └── topic_model.py      # GPT-driven granular topic extraction
├── cache/                  # Multi-level cache (Git ignored)
│   ├── calculation/        # Numeric preprocessing cache
│   ├── clusters/           # BERTopic result cache
│   ├── coverage/           # Metric calculation cache
│   ├── evaluations/        # GPT evaluation cache
│   ├── reports/            # Generated summary reports
│   ├── segments/           # Time segmentation cache
│   └── topics/             # Extracted topics cache
├── classification/         # Processed datasets
│   └── output_Classification.csv  # Main dataset (Managed via Git LFS)
├── preprocessing/          # Raw and intermediate data (Git ignored)
├── requirements.txt        # Python dependency list
└── .gitignore              # Repository exclusion rules
```

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.9+
- [Git LFS](https://git-lfs.com/) (Required for downloading the dataset)
- OpenAI API Key (For LLM features)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "06 app"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Git LFS**:
   ```bash
   git lfs pull
   ```

## 🖥️ Running the App

To launch the dashboard, run the following command from the project root:

```bash
streamlit run app/app.py
```

> [!TIP]
> Ensure you have your OpenAI API key ready. You can enter it directly in the sidebar of the application.

## 🔑 Cache Management

The application heavily utilizes caching to reduce API costs and improve performance. Caches are stored in the `cache/` directory. If you wish to force a recalculation for a specific user, you can manually delete the corresponding JSON file in the relevant subfolder.
