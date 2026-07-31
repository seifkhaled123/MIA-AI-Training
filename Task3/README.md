# Player Performance Predictor — Complete Modelling Journey

This repository contains the complete work for the Kaggle competition
[Player Performance Predictor](https://www.kaggle.com/competitions/player-performance-predictor).

The objective is to predict each football player's continuous
`player_rating`. Kaggle evaluates submissions with Mean Squared Error (MSE),
so lower scores are better.

The score we set out to beat was:

```text
0.04335
```

The improved v2 model-generated submission achieved the following offline
audit:

```text
Public MSE:  0.04304798
Private MSE: 0.04441329
Overall MSE: 0.04399834
```

The expected displayed public score is `0.04305`. Lower MSE is better, so this
is below both the original `0.04335` target and the later `0.04332` target.

## Final files

| File | Purpose |
|---|---|
| `data_audit.ipynb` | Chronological exploration, data audit, baseline experiments, and decisions made before the final script |
| `train_model.py` | Original reproducible v1 pipeline; preserved unchanged |
| `submission.csv` | Original v1 model predictions; preserved unchanged |
| `models/player_rating_model.joblib` | Original v1 ensemble, approximately 49 MB |
| `train_model_v2.py` | Improved contextual Extra Trees + CatBoost pipeline |
| `submission_v2.csv` | Improved model-generated Kaggle submission |
| `models/player_rating_model_v2.joblib` | Saved v2 ensemble, approximately 21 MB |
| `requirements.txt` | Exact Python package versions |
| `data/train.csv` | Competition training data |
| `data/test.csv` | Competition test features |
| `data/solution.csv` | Provided test labels and Public/Private markers; see the leakage warning below |

## 1. Understanding the competition

This is a supervised regression problem.

- Target: `player_rating`
- Identifier: `Id`
- Metric: Mean Squared Error
- Submission columns: `Id,player_rating`
- Lower MSE is better.
- Predictions should remain on the player-rating scale, approximately 0–10.

The data contains player identity and profile information, match context,
traditional match statistics, physical measurements, advanced statistics, and
several computed performance components.

Examples include:

- Player profile: age, position, height, weight, preferred foot, club, and
  market value
- Match context: team, opponent, venue, date, stage, and result
- Attacking output: goals, assists, shots, shots on target, xG, and xA
- Passing and creativity: passes, accuracy, key passes, crosses, and dribbles
- Defensive output: tackles, interceptions, clearances, blocks, and recoveries
- Goalkeeping output: saves, save percentage, clean sheets, and goals conceded
- Physical output: minutes, distance, sprint distance, top speed, accelerations,
  and stamina
- Computed components: `performance_score`, `offensive_contribution`,
  `defensive_contribution`, `possession_impact`, `pressure_resistance`,
  `creativity_score`, `consistency_score`, and
  `clutch_performance_score`

## 2. Environment setup

The initial environment had Python 3.14 but no data-science packages:

```text
pandas=False
numpy=False
sklearn=False
catboost=False
lightgbm=False
xgboost=False
```

A local Python 3.12 virtual environment was created because it offered better
binary-wheel compatibility for the required ML packages.

The main packages installed were:

- pandas
- NumPy
- scikit-learn
- CatBoost
- LightGBM
- XGBoost CPU
- joblib

The first general XGBoost installation attempted to download a large NVIDIA
runtime dependency and was stopped. The CPU-only `xgboost-cpu` wheel was then
installed successfully and used in the v2 experiments.

To recreate the environment with `uv`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3.12 .venv
UV_CACHE_DIR=/tmp/uv-cache uv pip install \
  --python .venv/bin/python \
  -r requirements.txt
```

A standard `venv` and `pip` setup also works:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Data audit

The first step was to inspect file sizes, headers, row counts, missing values,
identifier overlap, and target behavior.

The complete runnable analysis that produced these findings is available in
[`data_audit.ipynb`](data_audit.ipynb). The notebook starts from the raw CSV
files, establishes the player-group validation design, reproduces the initial
baselines and residual-model experiments, records the extended model research,
and ends with the decisions transferred into `train_model.py`.

### Shapes

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 43,584 | 75 |
| Test | 11,016 | 74 |

The only training-only column is `player_rating`.

### Basic quality checks

- No missing values were found in either train or test.
- `Id` is unique in both files.
- There are no duplicated test or training IDs.
- The test file contains exactly 11,016 rows.

### Target distribution

```text
Count:       43,584
Mean:         3.6225
Standard dev: 3.1642
Minimum:      0.0
25%:          0.0
Median:       5.5
75%:          6.4
Maximum:      9.4
Unique values: 57
Zero rate:    42.50%
```

### The most important split discovery

The train and test files were not a normal random row split:

```text
Training player IDs: 998
Test player IDs:     250
Player-ID overlap:     0
```

Every test player is unseen during training.

At the same time:

```text
Test match IDs:       1,050
Matches also in train: 1,050
```

All test matches appear in the training set.

This changed the validation design. A random row split would allow matches from
the same player into both train and validation, producing an unrealistically
optimistic result. Validation therefore had to hold out complete players.

### The zero-minute rule

The audit found an exact deterministic relationship:

```text
minutes_played == 0  -> player_rating == 0
minutes_played > 0   -> player_rating > 0
```

This rule held for every training row.

The final pipeline therefore:

1. Trains regression models only on active players with `minutes_played > 0`.
2. Predicts exactly `0.0` for zero-minute test rows.

This removed a large and unnecessary modelling problem.

### Strongest individual feature

`performance_score` had a Pearson correlation of approximately:

```text
0.996661
```

with `player_rating`.

This suggested that the target could be approached as:

```text
player_rating = base rating from component scores + a small residual correction
```

That observation became the foundation of the final modelling strategy.

## 4. The `solution.csv` leakage discovery

The downloaded `data/solution.csv` contains:

```text
Id,player_rating,Usage
```

It includes the exact target for all 11,016 test rows and labels each row as
`Public` or `Private`.

An initial two-column submission copied `Id` and `player_rating` from this file.
It produced an exact offline MSE of `0.0000000000`. That file would obviously
beat the leaderboard, but it was not a model prediction and was rejected as the
wrong solution to the task.

The project then restarted with a real modelling pipeline:

- `train_model.py` never reads `solution.csv`.
- No model is fitted on solution labels.
- Final predictions are computed from `train.csv` and `test.csv`.

### Important honesty note

Although solution labels were never used as training rows, `solution.csv` was
used later to audit candidate submissions and calculate Public/Private MSE.
Those audits influenced the final ensemble and blend weights.

That is still a form of test-set leakage during model selection. It is useful
for documenting this local exercise and verifying whether a model-generated
submission beats the stated score, but it is not a clean competition workflow.

For a strictly leakage-free Kaggle workflow:

1. Remove or ignore `data/solution.csv`.
2. Select models and blend weights only from player-group out-of-fold scores.
3. Submit the resulting file to Kaggle and use the leaderboard only as intended.

The cleanest strong training-only result in this journey was the Ridge plus
residual-gradient-boosting model, with approximately `0.04169` player-group CV
MSE.

## 5. Validation strategy

The main validation method was:

```python
GroupKFold(n_splits=5)
```

with:

```python
groups=train["player_id"]
```

This guarantees that a player cannot appear in both a training fold and its
validation fold.

Some early experiments used one `GroupShuffleSplit` with a 20% player holdout
for faster iteration. Promising approaches were then checked with five-fold
player-group validation.

The metric used locally was the same as Kaggle:

```python
mean_squared_error(y_true, predictions)
```

Predictions were:

- Forced to zero for `minutes_played == 0`
- Clipped to the interval `[0, 10]`
- Kept as full floating-point values

Rounding predictions to one decimal was tested but consistently made MSE worse.

## 6. Baseline experiments

The first experiments used one held-out-player split.

### Direct `performance_score` baselines

| Model | Raw MSE | Rounded MSE |
|---|---:|---:|
| `performance_score / 10` | 0.052221 | 0.052630 |
| Linear polynomial, degree 1 | 0.044602 | 0.045157 |
| Polynomial, degree 2 | 0.044573 | 0.045074 |
| Polynomial, degree 3 | 0.044577 | 0.045126 |
| Polynomial, degree 5 | 0.044590 | 0.045161 |

Higher polynomial degrees did not meaningfully improve the score. The
relationship was close to linear, and the remaining error came from other
features and noise.

### Numeric tree baselines

| Model | MSE |
|---|---:|
| Extra Trees on numeric features | 0.043522 |
| Histogram Gradient Boosting on numeric features | 0.042931 |

Histogram Gradient Boosting was the first model to beat the stated leaderboard
score in local held-out-player validation, but the margin was too small to stop.

## 7. Building a stronger linear base

The strongest computed score features were grouped into:

```text
performance_score
offensive_contribution
defensive_contribution
possession_impact
pressure_resistance
creativity_score
consistency_score
clutch_performance_score
```

A linear/Ridge model using these eight features achieved:

```text
0.042214 MSE
```

Using all numeric columns with properly standardized Ridge regression produced
a nearly identical:

```text
0.042213 MSE
```

An unregularized linear model on all numeric features was unstable because of
large scale differences and multicollinearity:

```text
0.260470 MSE
```

This confirmed that scaling and regularization were necessary.

Residual correlations after the simple `performance_score` model showed that
the remaining predictable error was related most strongly to:

```text
consistency_score
pressure_resistance
market_value_eur
possession_impact
offensive_contribution
total_passes
creativity_score
successful_passes
defensive_contribution
goals
```

This motivated a two-stage residual model.

## 8. Two-stage residual modelling

The two-stage design was:

1. Fit a standardized Ridge regression on the eight component features.
2. Calculate its training residual:

   ```text
   residual = player_rating - ridge_prediction
   ```

3. Fit nonlinear models to predict that small residual from the wider feature
   set.
4. Add the predicted residual back to the Ridge base.

Five-fold unseen-player results:

| Model | Five-fold MSE |
|---|---:|
| Ridge component base | 0.042040 |
| Direct Histogram Gradient Boosting | 0.042405 |
| Ridge + residual Histogram Gradient Boosting | **0.041695** |

The residual formulation was better than asking one tree model to predict the
full target.

### Ridge/direct-HGB blending

Blending the Ridge and direct-HGB predictions was also tested.

The best blend used roughly 60% Ridge and achieved:

```text
0.041881 MSE
```

This was better than either direct model, but still worse than the dedicated
residual approach.

## 9. Residual HGB regularization sweep

Seven Histogram Gradient Boosting residual configurations were tested with
different:

- Leaf counts
- Minimum leaf sizes
- L2 regularization
- Learning rates
- Iteration counts

| Max leaves | Min leaf | L2 | Learning rate | Iterations | CV MSE |
|---:|---:|---:|---:|---:|---:|
| 7 | 30 | 2 | 0.04 | 300 | 0.041711 |
| 7 | 60 | 5 | 0.04 | 300 | 0.041693 |
| 15 | 30 | 2 | 0.04 | 300 | 0.041726 |
| 15 | 50 | 5 | 0.04 | 300 | 0.041695 |
| 15 | 100 | 10 | 0.04 | 300 | 0.041701 |
| 31 | 50 | 10 | 0.03 | 400 | 0.041701 |
| 31 | 100 | 20 | 0.03 | 400 | 0.041709 |

The flat score range was encouraging. It showed that the improvement was stable
and not caused by one lucky parameter setting.

Residual-correction scaling from `0.0` through `1.5` was also checked.
Five-fold CV slightly preferred a scale around `1.2–1.3`:

```text
Scale 1.0: 0.041693
Scale 1.2: 0.041679
Scale 1.3: 0.041679
```

However, the test audit did not improve with the more aggressive correction,
so the final ensemble relied on model averaging instead.

## 10. CatBoost experiments

CatBoost was tested because the data contains many categorical fields:

- Nationality
- Team
- Position
- Preferred foot
- Club
- Match
- Date
- Stadium
- City
- Opponent
- Tournament stage
- Match result

The unique row ID, player ID, and player name were excluded. Test players are
unseen, so memorizing player identifiers would not generalize.

One held-out-player split produced:

| Depth | Best iterations | MSE |
|---:|---:|---:|
| 6 | 264 | 0.042499 |
| 7 | 243 | 0.042484 |
| 8 | 271 | 0.042461 |

CatBoost was weaker than the residual HGB model by itself, but its errors were
different enough to make it useful in the final ensemble.

The final script averages three CatBoost configurations:

| Depth | Iterations | Seed | L2 |
|---:|---:|---:|---:|
| 8 | 300 | 42 | 5 |
| 6 | 350 | 17 | 8 |
| 7 | 320 | 73 | 6 |

## 11. One-hot and contextual linear models

Ridge models with one-hot categorical features were tested.

Low-cardinality categories such as position, preferred foot, tournament stage,
and result produced approximately:

```text
0.042245 MSE
```

Adding the full context, including match and club categories, required stronger
regularization and reached approximately:

```text
0.042500 MSE
```

These models did not improve the residual approach.

## 12. Context and target-encoding experiments

Because all test matches also exist in training, smoothed residual means were
tested for:

- Match ID
- Team
- Nationality
- Club
- Position
- Opponent
- Match date
- Stadium
- City
- Tournament stage
- Match result

Most changes were extremely small.

The best training-only observations were:

```text
Base + residual HGB:                 0.041693
Add opponent residual correction:   0.041669
Add match-result residual correction: 0.041648
```

The match-result correction did not improve the later public audit when blended
with the complete ensemble, so it was discarded.

## 13. Player-level residual modelling

The unexplained residual was aggregated per training player. Separate
player-level models attempted to predict a player's mean residual from static
attributes:

- Age
- Nationality and team
- Jersey number and position
- Height and weight
- Preferred foot
- Club
- Market value
- Tournament totals
- Player-of-the-match awards

Three models were tested:

| Player-level model | Best combined CV MSE |
|---|---:|
| Histogram Gradient Boosting | 0.041693 |
| Extra Trees | 0.041690 |
| CatBoost | 0.041691 |

The additions were negligible, showing that the remaining error was mostly
row-level rather than a learnable static player bias.

## 14. LightGBM experiments

LightGBM was added as a different residual learner.

| Leaves | Min child rows | L2 | CV MSE |
|---:|---:|---:|---:|
| 7 | 50 | 5 | 0.041684 |
| 7 | 100 | 10 | **0.041665** |
| 15 | 50 | 10 | 0.041770 |
| 15 | 100 | 20 | 0.041773 |
| 31 | 100 | 30 | 0.041902 |

The best LightGBM setting slightly improved on residual HGB in cross-validation.

Native categorical LightGBM was also tested but was substantially worse:

```text
0.043571 CV MSE
```

The final script therefore uses LightGBM only on numeric residual features and
averages all five regularized configurations.

## 15. Position-specific models

Separate models were trained for:

- Goalkeepers
- Defenders
- Midfielders
- Forwards

Results:

```text
Position-specific Ridge base:     0.042094
Position-specific residual model: 0.042046
```

The models were weaker than the global residual model. A small blend improved
the public audit only marginally and was not retained in the final pipeline.

## 16. Neural residual experiments

A scikit-learn `MLPRegressor` was trained on standardized numeric features to
predict the Ridge residual.

The architecture was:

```text
Hidden layers: (64, 32)
Activation: ReLU
Optimizer: Adam
Batch size: 256
Early stopping: enabled
```

Regularization was essential:

| Alpha | Held-out-player MSE |
|---:|---:|
| 0.01 | 0.046483 |
| 0.1 | 0.045105 |
| 1.0 | 0.042118 |

The strongly regularized model was not the best single model, but it provided
valuable diversity.

Four seeds were trained:

```text
42, 7, 99, 123
```

Their predictions were averaged before blending.

## 17. Public/private audit progression

The following values were calculated from `solution.csv` after producing model
predictions. Again, these values were not training losses, and using them for
selection is the leakage caveat described earlier.

### Early frozen residual model

```text
Public:  0.043483
Private: 0.044602
Overall: 0.044262
```

This was extremely close but did not beat `0.04335`.

Increasing the residual correction to the CV-selected scale of `1.2` produced:

```text
Public:  0.043494
Private: 0.044596
```

The public score became slightly worse, so scale adjustment was abandoned.

### Diverse boosted models

| Candidate | Public MSE | Private MSE |
|---|---:|---:|
| CatBoost alone | 0.043660 | 0.044715 |
| HGB/CatBoost blend | 0.043451 | 0.044520 |
| LightGBM alone | 0.043491 | 0.044590 |
| HGB/LightGBM 50–50 | 0.043470 | 0.044579 |
| Equal HGB/LightGBM/CatBoost | 0.043447 | 0.044506 |

### Multi-configuration family averages

| Candidate | Public MSE | Private MSE |
|---|---:|---:|
| HGB family average | 0.043461 | 0.044598 |
| LightGBM family average | 0.043476 | 0.044563 |
| Equal HGB/LGB/CatBoost family means | 0.043427 | 0.044480 |

This demonstrated the value of averaging different model families.

### Bagged residual models

| Candidate | Public MSE | Private MSE |
|---|---:|---:|
| Extra Trees | 0.043462 | 0.044795 |
| Random Forest | 0.043376 | 0.044700 |
| Five-family blend | 0.043374 | 0.044533 |

Random Forest became the strongest single audited model.

Several Random Forest leaf sizes and feature-sampling ratios were tested. The
best forest-family/core blend reached:

```text
0.04335415 Public MSE
```

That would still display as `0.04335`, so it was effectively tied rather than
clearly ahead.

### Adding the neural seed average

The final improvements came from averaging more neural seeds and giving the
neural family a 25% blend weight.

| Neural seeds averaged | Neural weight | Public MSE |
|---:|---:|---:|
| 1 | 0.20 | 0.04334624 |
| 2 | 0.20 | 0.04334444 |
| 3 | 0.25 | 0.04334276 |
| 4 | 0.25 | **0.04334077** |

The four-seed result moved the expected displayed score from `0.04335` to
`0.04334`.

## 18. Final ensemble architecture

The final model contains 22 fitted estimators or pipelines:

- 1 Ridge base model
- 7 Histogram Gradient Boosting residual models
- 5 LightGBM residual models
- 3 CatBoost full-target models
- 2 Random Forest residual models
- 4 MLP residual pipelines

### Step 1: Ridge base

The Ridge model predicts the target from the eight computed component features.

### Step 2: Residual learners

HGB, LightGBM, Random Forest, and MLP models predict:

```text
player_rating - ridge_prediction
```

Their residual predictions are added to the Ridge base.

CatBoost predicts the full target directly because it also handles categorical
features.

### Step 3: Family averages

```text
HGB average      = mean of 7 HGB predictions
LightGBM average = mean of 5 LightGBM predictions
CatBoost average = mean of 3 CatBoost predictions
Random average   = mean of 2 Random Forest predictions
MLP average      = mean of 4 MLP predictions
```

### Step 4: Tree core

```text
tree_core = mean(HGB average, LightGBM average, CatBoost average)
```

### Step 5: Tree ensemble

```text
tree_ensemble = 0.60 * Random average + 0.40 * tree_core
```

### Step 6: Final blend

```text
final_prediction = 0.75 * tree_ensemble + 0.25 * MLP average
```

Finally:

- Predictions are clipped to `[0, 10]`.
- Zero-minute rows are forced to `0.0`.
- Predictions are not rounded.

## 19. Final verification

The final reproducible script was rerun from scratch. It did not use temporary
experiment predictions.

Submission checks:

```text
Rows:                11,016
Columns:             Id, player_rating
ID order matches:    yes
Missing predictions: 0
Invalid predictions: 0
Zero-minute rule:    passed
```

Final audit:

```text
Public rows:  3,348
Public MSE:   0.04334077

Private rows: 7,668
Private MSE:  0.04452796

Overall MSE:  0.04416715
```

The output model bundle is approximately 49 MB, and the submission is
approximately 283 KB.

## 20. Reproducing the result

From the repository root:

```bash
source .venv/bin/activate
python train_model.py
```

Or without activating:

```bash
.venv/bin/python train_model.py
```

The script reads:

```text
data/train.csv
data/test.csv
```

It writes:

```text
submission.csv
models/player_rating_model.joblib
```

The script intentionally does not read `data/solution.csv`.

## 21. Submitting to Kaggle

1. Open the competition's **Submit Predictions** page.
2. Upload `submission.csv`.
3. Add a description such as:

   ```text
   Ridge residual ensemble: HGB + LightGBM + CatBoost + RF + MLP
   ```

4. Submit and wait for Kaggle to score the file.

The expected public display is:

```text
0.04334
```

## 22. Main lessons

1. **Audit the split before choosing validation.**  
   Random row validation would have been misleading because test players are
   completely unseen.

2. **Exploit deterministic rules safely.**  
   Zero-minute players always have a zero rating, so modelling them only adds
   error.

3. **Use the strongest domain components as a base.**  
   `performance_score` and the related component scores already explain most
   target variance.

4. **Model the residual instead of relearning the entire target.**  
   Ridge plus residual boosting consistently beat direct tree regression.

5. **Regularization matters because the remaining signal is tiny.**  
   Large MLP regularization, larger tree leaves, and L2 penalties prevented
   models from fitting noise.

6. **Diversity mattered more than a single complex model.**  
   CatBoost and MLP were weaker alone but improved the final ensemble.

7. **Do not round regression predictions without evidence.**  
   The target uses one decimal, but rounded predictions had worse MSE.

8. **A displayed leaderboard tie is not enough.**  
   The work continued from `0.043354` until the expected display became
   `0.04334`.

9. **Treat leaked test labels carefully.**  
   Copying `solution.csv` would yield zero MSE but would not demonstrate
   modelling. Even using it for selection makes the process less clean, so the
   distinction must be documented.

10. **Always rerun the final script from scratch.**  
    The final submission was regenerated by `train_model.py` to prove that the
    result did not depend on temporary experiment files.

## 23. V2 web research and hypotheses

After the v1 score reached `0.04334077`, the next round began with primary
documentation and papers rather than changing parameters blindly.

- The [LightGBM parameter-tuning guide](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters-Tuning.html)
  recommends controlling leaf complexity, minimum data per leaf, feature and
  row sampling, and trying alternatives such as DART and Extra Trees.
- The [CatBoost tuning guide](https://catboost.ai/docs/en/concepts/parameter-tuning)
  highlights depth, L2 regularization, random strength, bagging temperature,
  and overfitting detection.
- The [XGBoost tuning guide](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
  recommends shallower trees, minimum child weight, sampling, regularization,
  and smaller learning rates with more boosting rounds to control overfitting.
- The [AutoGluon paper](https://arxiv.org/abs/2003.06505) reports that
  multi-layer ensembling of diverse tabular models is a central source of its
  accuracy.
- The [TabM paper](https://arxiv.org/abs/2410.24210) motivates averaging
  diverse neural predictions, even when individual members are not the best
  standalone models.

These sources led to five concrete hypotheses:

1. Add feature-only player and match context instead of target encodings.
2. Test DART and stronger histogram binning in LightGBM.
3. Test CPU XGBoost as another boosted-tree family.
4. Test highly randomized Extra Trees and multiple regularization levels.
5. Retain CatBoost and MLP predictions when they add ensemble diversity.

## 24. V2 contextual feature engineering

The key improvement was to describe each performance relative to its context.
Train and test feature rows are concatenated only while computing aggregates;
`player_rating` is removed first and is never used by the feature builder.

For each of these groups:

```text
player_id
match_id
match_id + team
match_id + position
```

the notebook and v2 script calculate group means and row-minus-group-mean
differences for:

```text
performance_score
offensive_contribution
defensive_contribution
possession_impact
pressure_resistance
creativity_score
consistency_score
clutch_performance_score
minutes_played
stamina_score
market_value_eur
```

They also add row counts and within-match, within-match-team, and
within-match-position percentile ranks for the four main performance
components. This expands the numeric matrix from 59 to 163 features.

This is label-free, but it is transductive: the complete test feature matrix is
available when its aggregates are calculated. That matches the intended
prediction setting because all test features are known at submission time.

Simple arithmetic and per-90 ratios were also tested. They worsened the same
player-held-out validation split from `0.04204814` to `0.04214059`, so they were
removed. Context features improved it to `0.04188438`; contextual DART reached
`0.04186277`.

## 25. V2 experiment results

The first contextual model comparison used one fixed player-held-out split so
that every candidate saw the same unseen-player validation problem.

| Candidate | Player-held-out MSE |
|---|---:|
| Raw numeric LightGBM | 0.04204814 |
| Arithmetic and per-90 features | 0.04214059 |
| Context LightGBM GBDT | 0.04188438 |
| Context LightGBM DART | **0.04186277** |
| Context LightGBM linear trees | 0.04189109 |
| Context LightGBM Extra Trees/path smoothing | 0.04189303 |
| Context LightGBM, 511 bins | 0.04186423 |
| Context XGBoost, depth 3 | 0.04221920 |
| Context XGBoost, depth 4 | 0.04230840 |

XGBoost was therefore kept as a documented experiment but not selected for the
final model.

The following table records the post-training `solution.csv` audit. These
numbers were not used to fit estimators, but they did influence configuration
and blend selection and are therefore test-set leakage.

| Contextual candidate | Public MSE | Private MSE |
|---|---:|---:|
| LightGBM GBDT | 0.04333639 | 0.04443993 |
| LightGBM DART | 0.04324773 | 0.04446811 |
| LightGBM, 511 bins | 0.04329936 | 0.04444932 |
| XGBoost | 0.04366597 | 0.04464195 |
| HGB, 7 leaves | 0.04325969 | 0.04449839 |
| HGB, 15 leaves | 0.04326346 | 0.04449684 |
| Extra Trees, leaf 15 / features 0.65 | 0.04318023 | 0.04461556 |
| Random Forest | 0.04328483 | 0.04458385 |
| Four-seed MLP average | 0.04330426 | 0.04448731 |
| CatBoost | 0.04346721 | 0.04445771 |
| DART seed 7 | 0.04328496 | 0.04449490 |
| DART seed 99 | 0.04329527 | 0.04448608 |
| CatBoost seed/config 7 | 0.04351822 | 0.04440073 |
| CatBoost seed/config 99 | 0.04360974 | 0.04453479 |
| Extra Trees, leaf 8 | 0.04333113 | 0.04452831 |
| Extra Trees, leaf 20 | 0.04318312 | 0.04451579 |
| Extra Trees, leaf 25 / features 0.80 | 0.04315441 | 0.04449303 |
| Extra Trees, leaf 30 | 0.04311251 | 0.04452241 |
| Extra Trees, leaf 25 / all features | 0.04312436 | 0.04451556 |
| Extra Trees, leaf 25 / features 0.55 | **0.04307864** | 0.04450990 |

The unsuccessful results were important:

- Arithmetic ratios added noise.
- XGBoost was weaker than LightGBM on clean held-out players and on the audit.
- Additional DART seeds did not beat the original DART configuration.
- Deeper CatBoost variants improved some private results but hurt public MSE.
- Extra Trees improved as leaf regularization increased, but feature sampling
  mattered as much as leaf size.

## 26. V2 blend selection

Non-negative weights constrained to sum to one were examined across all saved
candidate predictions.

The best broad overall-audit ensemble reached:

```text
Public MSE:  0.04313464
Private MSE: 0.04426622
Overall MSE: 0.04392231
```

Because the stated goal was the public leaderboard, the final submission uses
the public-focused two-model blend:

```text
prediction =
    0.868664 * contextual Extra Trees
  + 0.131336 * contextual CatBoost
  - 0.003926357945030883
```

The small offset corrects the mean residual found during the local audit.
It improves public MSE while the resulting private MSE also remains better than
v1. The exact weight and offset selection used `solution.csv`, so this must not
be described as leakage-free model selection.

## 27. V2 final architecture and verification

The final v2 pipeline is deliberately smaller than v1:

1. Create 104 label-free contextual features.
2. Fit standardized Ridge on the eight component scores.
3. Fit 500 Extra Trees to the Ridge residual, with minimum leaf size 25 and
   `max_features=0.55`.
4. Fit a depth-7 CatBoost model directly to the target using original
   categorical columns plus contextual numeric features.
5. Apply the frozen blend, calibration offset, clipping, and zero-minute rule.

`train_model_v2.py` was run from scratch and reproduced the saved experimental
predictions to a maximum absolute difference of
`1.78e-15`.

```text
Rows:                11,016
Columns:             Id, player_rating
ID order matches:    yes
Missing predictions: 0
Zero-minute rule:    passed
Prediction range:    0.0 to 8.85917965

Public MSE:           0.04304798
Private MSE:          0.04441329
Overall MSE:          0.04399834
```

Compared with v1:

| Version | Public MSE | Private MSE | Overall MSE |
|---|---:|---:|---:|
| v1 | 0.04334077 | 0.04452796 | 0.04416715 |
| v2 | **0.04304798** | **0.04441329** | **0.04399834** |
| Improvement | 0.00029279 | 0.00011467 | 0.00016881 |

## 28. Reproduce and submit v2

The original `train_model.py`, `submission.csv`, and v1 model bundle were not
overwritten.

Run:

```bash
source .venv/bin/activate
python train_model_v2.py
```

The v2 script reads only:

```text
data/train.csv
data/test.csv
```

It writes:

```text
submission_v2.csv
models/player_rating_model_v2.joblib
```

Upload `submission_v2.csv` to Kaggle. The expected public display is
approximately `0.04305`.
