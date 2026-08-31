# Dixon-Coles Engine Methodology

## Scope

This document defines the current production Dixon-Coles pricing methodology used by:

- `docs/win/soccer/scripts/02_juice/apply_juice.py`

League configuration is read from:

- `docs/win/soccer/config/juice/epl/dc_soccer_pricing_engine.csv`
- `docs/win/soccer/config/juice/bundesliga/dc_soccer_pricing_engine.csv`
- `docs/win/soccer/config/juice/la_liga/dc_soccer_pricing_engine.csv`
- `docs/win/soccer/config/juice/ligue1/dc_soccer_pricing_engine.csv`
- `docs/win/soccer/config/juice/mls/dc_soccer_pricing_engine.csv`
- `docs/win/soccer/config/juice/serie_a/dc_soccer_pricing_engine.csv`

The original source/generator that produced the historical engine CSV grids is not available in the current repository and its external location is unknown.

Production pricing no longer uses nearest-neighbor lookup against those stored grids. `apply_juice.py` reads the league `rho` value from the engine CSV and calculates Dixon-Coles probabilities directly at runtime from each match's exact `home_xg` and `away_xg`.

## Production authority

`apply_juice.py` is the single production pricing authority.

For production probability and fair-odds fields:

1. Exact match xG values are validated.
2. The league `rho` value is loaded from `dc_soccer_pricing_engine.csv`.
3. Dixon-Coles probabilities are calculated at runtime from the exact xG values.
4. Fair decimal odds are calculated as the reciprocal of those probabilities.
5. Runtime mathematical invariants are validated before output.

No nearest-neighbor xG approximation is used for production pricing.

## Base model

For home expected goals `lambda_home` and away expected goals `lambda_away`, the unadjusted score probabilities are independent Poisson:

`P(X=x, Y=y) = Pois(x; lambda_home) * Pois(y; lambda_away)`

where:

`Pois(k; lambda) = exp(-lambda) * lambda^k / k!`

## Dixon-Coles rho correction

The runtime engine uses the standard Dixon-Coles low-score adjustment with the league's stored `rho`.

For a score `(x,y)`, multiply the independent-Poisson probability by `tau(x,y)`:

- `(0,0)`: `1 - lambda_home * lambda_away * rho`
- `(0,1)`: `1 + lambda_home * rho`
- `(1,0)`: `1 + lambda_away * rho`
- `(1,1)`: `1 - rho`
- all other scores: `1`

Thus:

`P_DC(x,y) = P_Poisson(x,y) * tau(x,y)`

The resulting score-grid probability mass is normalized before market probabilities are returned.

## Equivalent market-level correction

Let:

`delta = rho * lambda_home * lambda_away * exp(-(lambda_home + lambda_away))`

Relative to independent Poisson:

- `home_win_DC = home_win_Poisson + delta`
- `draw_DC = draw_Poisson - 2 * delta`
- `away_win_DC = away_win_Poisson + delta`
- `btts_yes_DC = btts_yes_Poisson - delta`
- `btts_no_DC = 1 - btts_yes_DC`

For negative `rho`, `delta` is negative, so home-win and away-win probability decrease, draw probability increases, and BTTS Yes increases.

## Totals

The Dixon-Coles correction only redistributes probability among `0-0`, `0-1`, `1-0`, and `1-1`.

The net probability change across those cells is zero, and all four scores are Under 2.5 and Under 3.5. Therefore the correction does not change:

- Over/Under 2.5
- Over/Under 3.5

## Fair odds

Fair decimal odds are calculated directly from the runtime corrected probability:

`fair_decimal = 1 / probability`

Accordingly:

- `engine_home_fair_decimal` corresponds to runtime `engine_home_prob`
- `engine_draw_fair_decimal` corresponds to runtime `engine_draw_prob`
- `engine_away_fair_decimal` corresponds to runtime `engine_away_prob`
- totals and BTTS fair decimal fields likewise correspond to their runtime engine probabilities

## Runtime validation

Production pricing validates, at minimum:

1. Required xG inputs are present, numeric, finite, and non-negative.
2. `expected_total_goals` agrees with `home_xg + away_xg` within configured tolerance.
3. Home + Draw + Away sums to approximately 1.
4. Over 2.5 + Under 2.5 sums to approximately 1.
5. Over 3.5 + Under 3.5 sums to approximately 1.
6. BTTS Yes + BTTS No sums to approximately 1.
7. Every probability is finite and between 0 and 1.
8. Fair decimal odds equal `1 / probability` within numerical tolerance.
9. Swapping home and away xG swaps home/away win probabilities while preserving symmetric markets within tolerance.

Invalid rows cannot silently enter production pricing.

## Historical engine-grid status

The `dc_soccer_pricing_engine.csv` files are retained as league configuration/history, but current production pricing does not copy their precomputed probability rows.

The current runtime consumer uses the league `rho` value from those files and prices the exact xG pair itself.

Because the original generator/source for the historical grids was not retained, those historical grids cannot be independently regenerated from repository artifacts alone.

## Legacy 3-way juice is separate

The league `3way_juice.csv` files are not the production Dixon-Coles probability source.

They are still used by `apply_juice.py` to create legacy diagnostic 1X2 adjustment fields. Those diagnostic fields are separate from the authoritative `engine_*` production probabilities and fair odds.

See:

- `docs/win/soccer/config/juice/Explanation.txt`
- `docs/win/soccer/config/juice/method.txt`
