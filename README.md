LIVE WEBSITE URL: Replace this line with your deployed Streamlit URL before submission.

# Octagon Analytics - DATA 3301 Final Project Website

This Streamlit website is the final project report. It includes project introduction, data provenance and EDA, five predictive questions, held-out model comparisons, a hypothetical matchup predictor, an exploratory YOLO pose-analysis page, decisions/reflection, limitations, and a submission checklist.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Files and order

1. `app.py` is the website entry point and report narrative. Replace `TEAM_MEMBERS` before submission.
2. `modeling.py` loads the supplied CSVs, creates leakage-aware pre-fight features, evaluates five questions on a chronological holdout, and powers the matchup tool.
3. `video_analysis.py` runs exploratory YOLO pose analysis for uploaded clips. It is intentionally separate from the outcome models.
4. `merged_stats_n_scorecards.csv` and `fighters_raw.csv` are required for the site. The remaining CSVs and notebooks document source data and exploratory work.

## Submission

Submit a single outer ZIP containing this website folder and a `code_and_data/` folder with all notebooks, Python files, README, and needed data. Leave the live deployment URL on the first line of this README.
