## Soccer ML model-selection history — future reference

Each league was **tested independently**, using its own historical match data. We did **not** choose one universal soccer algorithm. We treated **1X2, Over 2.5, Over 3.5, BTTS and goals as separate modeling problems**, compared candidate ML approaches, and promoted the selected production-compatible model for each league/market. Separate **predictability** and **skip** models were also retained so the betting selector can reject games where the trained model considers the market less reliable.

The final promoted models now in the repository are:

| League         | 1X2                          | Extra Draw model        | O2.5                   | O3.5                    | BTTS                    | Goals                       |
| -------------- | ---------------------------- | ----------------------- | ---------------------- | ----------------------- | ----------------------- | --------------------------- |
| **EPL**        | Extra Trees                  | —                       | LightGBM               | XGBoost **calibrated**  | Random Forest           | LightGBM                    |
| **Bundesliga** | CatBoost                     | —                       | Logistic               | XGBoost **calibrated**  | CatBoost                | CatBoost                    |
| **La Liga**    | Extra Trees                  | CatBoost **calibrated** | Logistic               | Logistic                | CatBoost                | CatBoost + Poisson          |
| **Serie A**    | Logistic                     | Logistic **calibrated** | Extra Trees            | Extra Trees             | Logistic                | Random Forest + CatBoost    |
| **Ligue 1**    | Logistic **calibrated**      | —                       | XGBoost **calibrated** | Extra Trees             | LightGBM **calibrated** | CatBoost + Extra Trees      |
| **MLS**        | Random Forest **calibrated** | Logistic                | Logistic               | CatBoost **calibrated** | XGBoost                 | Extra Trees + Random Forest |

Those choices are explicitly encoded in each league's `*_ml_infer.py`, not inferred from model filenames.

### How the test results were put into production

The winning models were saved under:

```text
docs/win/soccer/ml/<league>/models/<target>/production-compatible/
```

and each league received its own inference script under:

```text
docs/win/soccer/scripts/01_merge/<league>_ml_infer.py
```

The production pipeline was then changed to:

```text
merge
→ league ML inference
→ ml_* probabilities / goals / predictability / skip outputs
→ apply_juice converts those into authoritative engine_* probabilities
→ build_edges calculates fair odds / EV / Kelly / edge
→ soccer_select_bets applies markets.yaml filters plus predictability/skip filters
→ grading and reports
```

All six leagues are now explicitly treated as ML pricing leagues by `apply_juice.py`.

### Important note for future work

The repository preserves the **final selected models and exactly how they are used in production**, but it does **not appear to preserve the complete original candidate leaderboard/test-metric tables for every league**. Therefore a future coder/AI should **not invent a reason that one algorithm beat another** or replace a model because another algorithm is generally considered better. The current models are the results of the completed league-by-league testing process; changing them should require **retesting that league/market first**.
