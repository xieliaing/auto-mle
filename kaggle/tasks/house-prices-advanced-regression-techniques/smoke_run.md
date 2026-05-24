# Smoke Test Run: House Prices - Advanced Regression Techniques
Date       : 2026-05-24
Python     : 3.11.14  (CPython)
Platform   : Windows-10-10.0.26200-SP0
scikit-learn   : 1.8.0
pandas         : 3.0.1
numpy          : 2.4.3
Git commit : 2b17c76

## Output

[ok] data loading - train=500 val=292 features=81
[ok] model training - RMSLE=0.1201 n_eval=292 predictions_df shape=(292, 7)
[ok] second iteration - RMSLE=0.1208
[ok] sampler - population=292 sampled=200 strategy=stratified_score_band bands: near=185 mid=103 far=4
  [defect] population=292 sampled=200 strategy=stratified_score_band has_confidence=True
  [defect] bands - near: 185 rows / 100 budget  mid: 103 / 60  far: 4 / 40
Defect analysis: iter1 -> iter2
  Population / sample : 292 / 200  (strategy=stratified_score_band)
  Score bands         : near: 185 rows (budget 100) | mid: 103 rows (budget 60) | far: 4 rows (budget 40)
  Fixed (wrong->right): 9 (4.5%)
  New errors          : 3 (1.5%)
  Persistent errors   : 17 (8.5%)
  Net change          : +6
  Fixed confusion transitions (top 3):
    * true=0 | prev=1 -> curr=0: x5
    * true=1 | prev=0 -> curr=1: x4
  New error transitions (top 3):
    * true=0 | prev=0 -> curr=1: x3
[ok] defect analysis - fixed=9 new_errors=3 persistent=17 net=+6
[ok] learning journal - 2 entries, format_for_proposer works

==================================================
Results: 6 passed, 0 failed

## Exit code: 0
