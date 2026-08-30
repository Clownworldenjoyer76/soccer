# Dixon-Coles Engine Methodology

## Scope

This document defines the mathematics represented by the six files:

- `epl/dc_soccer_pricing_engine.csv`
- `bundesliga/dc_soccer_pricing_engine.csv`
- `la_liga/dc_soccer_pricing_engine.csv`
- `ligue1/dc_soccer_pricing_engine.csv`
- `mls/dc_soccer_pricing_engine.csv`
- `serie_a/dc_soccer_pricing_engine.csv`

Consumer:

- `docs/win/soccer/scripts/02_juice/apply_juice.py`

The original source/generator that produced the engine CSVs is not available in the current repository and its external location is unknown. The calculation methodology is therefore documented here so the stored model is explicit and reproducible.

## Base model

For home expected goals `lambda_home` and away expected goals `lambda_away`, the unadjusted score probabilities are independent Poisson:

`P(X=x, Y=y) = Pois(x; lambda_home) * Pois(y; lambda_away)`

where:

`Pois(k; lambda) = exp(-lambda) * lambda^k / k!`

## Dixon-Coles rho correction

The engine uses the standard Dixon-Coles low-score adjustment with the row's stored `rho`.

For a score `(x,y)`, multiply the independent-Poisson probability by `tau(x,y)`:

- `(0,0)`: `1 - lambda_home * lambda_away * rho`
- `(0,1)`: `1 + lambda_home * rho`
- `(1,0)`: `1 + lambda_away * rho`
- `(1,1)`: `1 - rho`
- all other scores: `1`

Thus:

`P_DC(x,y) = P_Poisson(x,y) * tau(x,y)`

The current engine rows use `rho = -0.1`.

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

The existing totals probabilities remain valid when rho is applied.

## Fair odds

Fair decimal odds are the reciprocal of the corrected probability:

`fair_decimal = 1 / probability`

Accordingly:

- `home_win_fair_odds` must correspond to corrected `home_win`
- `draw_fair_odds` must correspond to corrected `draw`
- `away_win_fair_odds` must correspond to corrected `away_win`
- `btts_yes_fair_odds` must correspond to corrected `btts_yes`
- `btts_no_fair_odds` must correspond to corrected `btts_no`

## Consumer behavior

`apply_juice.py` does not calculate Dixon-Coles probabilities at runtime. It selects the nearest engine row in `(lambda_home, lambda_away)` space and copies the precomputed market probabilities and fair odds from the CSV.

Therefore `rho` must already have been applied when the engine CSV is generated.

## Verification performed

The engine audit found:

- EPL, Bundesliga, La Liga, and Ligue 1 previously stored `rho = -0.1` while their market probabilities reproduced independent Poisson.
- MLS and Serie A already reflected the Dixon-Coles rho correction.
- EPL, Bundesliga, La Liga, and Ligue 1 were corrected so the stored probabilities now reflect `rho = -0.1`.
- MLS and Serie A were left unchanged with respect to rho because they already applied it.

After regeneration/correction, a non-zero rho measurably changes the affected 1X2 and BTTS probabilities.

Example validation for Liverpool vs Nottingham Forest:

Before rho correction:

- Home: `0.5748337181`
- Draw: `0.2136001194`
- Away: `0.2115165389`

After rho correction:

- Home: `0.5650397249`
- Draw: `0.2331881057`
- Away: `0.2017225457`

This confirms that the stored model now behaves as Dixon-Coles rather than independent Poisson for the rho-sensitive markets.

## Invariants

Future engine generation must satisfy all of the following:

1. Changing a non-zero `rho` changes the affected low-score probabilities.
2. Changing `rho` changes 1X2 and BTTS probabilities when lambdas are positive.
3. Home + Draw + Away remains approximately 1.0, subject only to any existing score-grid tail truncation.
4. Fair odds remain the reciprocal of their labeled probabilities.
5. Swapping `lambda_home` and `lambda_away` swaps home/away win probabilities while leaving draw unchanged within numerical tolerance.
6. The `rho` column must not be retained if the engine is ever changed back to independent Poisson.
