# Transformer Failure Prediction

## Business Problem
Predict transformer failures in oil & gas facilities 2-4 weeks in advance to enable planned maintenance instead of emergency repairs.

**Business Impact**: £4.2M annual savings validated over 18 months in production.

## Architecture
- **Processing**: Batch pipelines (daily scoring, weekly retraining)
- **Data Sources**: SCADA sensor readings + maintenance event logs
- **Models**: XGBoost classifier (failure prediction) + regressor (time-to-failure)
- **Deployment**: On-premise Linux server, cron-scheduled
- **Output**: Risk reports to Power BI dashboard for operations team

## Project Structure
```
src/          - Reusable Python modules
scripts/      - Executable batch jobs (what cron runs)
notebooks/    - Exploratory analysis (not deployed)
config/       - Configuration files
data/         - Data storage (not in git)
```

## How to Run

### Setup
```bash
pip install -r requirements.txt
pip install -e .
```

### Run Pipelines
```bash
# Weekly: Prepare training data
python scripts/run_feature_pipeline.py

# Weekly: Retrain models
python scripts/run_training_pipeline.py

# Daily: Generate predictions
python scripts/run_scoring_pipeline.py
```

### View Experiments
```bash
mlflow ui
# Access at http://localhost:5000
```

## Business Metrics
- **Cost-weighted optimization**: F-beta (β=2.88) based on £500k failure vs £60k false alarm
- **Production results**: 88% recall, 72% precision
- **ROI**: £4.2M annual savings, 3-week payback

## Technical Approach
- Domain-driven feature engineering (operations insights → features)
- Cost-aware model optimization (not accuracy maximization)
- Production-quality code (modular, config-driven, tested)
