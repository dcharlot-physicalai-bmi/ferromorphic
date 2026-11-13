# Where 40.803375080880265 comes from

The number is a pasted comment. At NeuroBench `e521c28f5798cb40bb9423758bf066dad53fe070` (tag 2.3.0) it occurs once, in `examples/mackey_glass/esn_benchmark.py` line 96. The executable script at that commit sets `repeat = 30` and loops `tau` from 17 through 30. The comment is not the output of the script as checked in.

`git blame` at the pin attributes the number to `4b7a23379dcdb320c07318825bb7b90606c31c67`, Jason Yik, 2023-10-07, message "update esn load mg". That commit's executable settings are the ones the comment belongs to:

- `repeat = 1`
- `single_series = False`, so every row of `neurobench/datasets/mackey_glass_parameters.csv` (tau 17 through 30)
- `seed_id = repeat_id`, which is 0
- `start_offset_range = arange(0, 0.5 * repeat, 0.5) * 75`, which is the single offset 0
- `warmup = 0.6` Lyapunov times, `warmup_pts = round(0.6 * pts_per_lyaptime)`, labels sliced after that warmup, then `esn.fit(...)`. The readout is fit on that run. It is not loaded from `model_data/esn.pth`
- hyperparameters from `echo_state_network_hyperparameters.csv`. That blob is `90b940a3197554da296063dc2796174819e80ec8` both in this commit and at the pin (the pin only moves the file to `examples/mackey_glass/model_data/`)
- the series path is `data/mackey_glass/mg_{tau}.npy`. Those files are not in the commit

`ea4207f3d07203d36512f71a5c7c38361df3e53a` (Jason Yik, 2023-10-09, "add MG results") is the commit that labels the same three lines "Score for repeat=1, single_series=False". In the same diff it changes the executable script to `single_series = True` and `repeat = 30`, and adds a different pasted mean, 18.382620390575333, for that new setting. `68ef2a88` later shortens the label above 40.803375080880265 to "Score for repeat=1" and leaves the number.

`poetry.lock` at `4b7a2337` and at `ea4207f3d072` pins torch 2.0.1. It does not say CPU or CUDA. The commit does not save per-tau scores, a machine, or a log. The three comment lines are the print statements' text, pasted back into the file.

sMAPE at `4b7a2337` is `neurobench/benchmarks/data_metrics.py`: `200 * mean(|preds - data[1]| / (|preds| + |data[1]|))`, then `nan_to_num(..., nan=200)`. At the pin the same expression is `neurobench/metrics/workload/smape.py`, class `SMAPE`, and `docs/metrics/workload_metrics/smape.rst` prints that formula with range [0, 200]. No paper or supplement is in the pin. `leaderboard.rst` names Mackey-Glass and sMAPE and does not print 40.803375080880265. `MG_tutorial.ipynb` does not contain it. GitHub code search finds only the comment. Issue and pull-request search for the number returns no hits. Issue 71 states the metric rule (sMAPE, NaN becomes 200) and does not state this score.

What is not in the commit, so an exact rerun of the pasted float is not specified:

- the bytes of `data/mackey_glass/mg_{tau}.npy`
- the torch build (CPU or CUDA) and the machine
- the per-tau scores that were averaged

The hyperparameter table, the tau list, the single seed 0, the zero offset, the 0.6 Lyapunov warmup, a fresh ridge fit at the CSV `ridge_param`, and the sMAPE formula are specified. The series files and the torch wheel are not. This note does not rerun the paste: the series files it names are not in the commit, so there is no byte string here to match 40.803375080880265 against.
