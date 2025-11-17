# Operations Guide

This project now ships with first-class tooling for running the AI pipeline in production. Use this document as a quick reference for recurring jobs and health checks.

## 1. Outcome Ingestion

The pipeline learns continuously from real match outcomes. Append rows to `ai_system/data/match_outcomes.csv` (or use `OutcomeManager.record_outcome`) with:

```
match_id, outcome, timestamp
2025-11-30-inter-juve,1,1732982400
```

Then schedule the batch job (hourly or daily):

```bash
python scripts/run_outcome_ingest.py            # uses default TTL
python scripts/run_outcome_ingest.py --ttl-hours 24  # reapply last 24h
```

This command will:

1. Skip outcomes already applied in the last `outcome_ttl_hours`.
2. Validate that the match exists in the meta feature store.
3. Call `AIPipeline.register_outcome` so the meta-layer updates its reliability.

## 2. Health Monitoring & Alerts

Run the health watcher after ingestion or on a cron schedule (e.g. every hour):

```bash
# Print report, fail if any alert is present
python scripts/meta_alerts.py --limit 200 --fail-on-alert --print-summary

# Optional: send to Slack/Teams and trigger the playbook
python scripts/meta_alerts.py \
  --webhook-url "$META_ALERT_WEBHOOK" \
  --apply-playbook \
  --fail-on-alert
```

### Playbook Configuration

Define mappings from alert codes to shell commands in `meta_playbook.json`. An example template is stored in `meta_playbook.example.json`:

```json
{
  "low_outcome_feedback": [
    "python scripts/run_outcome_ingest.py"
  ],
  "high_probability_rmse": [
    "python scripts/run_outcome_ingest.py --ttl-hours 24",
    "python training/update_models.py --target calibrator"
  ],
  "low_weight_xgboost": [
    "python training/update_models.py --target xgboost"
  ]
}
```

Copy the template, customize the commands, and set `alert_playbook_path` in `AIConfig` if you place it elsewhere.

## 3. Dashboards & Reporting

Generate an on-demand HTML summary (great for attaching to e-mails or Slack threads):

```bash
python scripts/meta_dashboard.py --output meta_report.html --limit 300
```

The report includes:

- Overall meta health (entries, exploration rate, RMSE, outcome ratio)
- Multi-window summaries (last 50 / last 200 / full history)
- Reliability snapshot per model
- Data quality metrics (avg data availability, historical coverage)
- Active alerts with recommended actions

You can host the resulting HTML on an internal portal or attach it to automated status updates.

## 4. Suggested Scheduling

| Interval | Command | Purpose |
|----------|---------|---------|
| Every 30–60 minutes | `scripts/run_outcome_ingest.py` | Apply new match outcomes |
| Hourly / Post-analysis | `scripts/meta_alerts.py --apply-playbook --fail-on-alert` | Detect issues and trigger mitigation |
| Daily | `scripts/meta_dashboard.py` | Publish summary report |

Adjust the cadence to match your production volume. All scripts are stateless batch jobs; there is no need to keep them running continuously.
