# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official implementation of **CompDiffuser** ("Generative Trajectory Stitching through Diffusion Composition", arXiv 2503.05153). A short-horizon diffusion planner is trained on OGBench `*-stitch` / `*-explore` datasets. At test time, several short trajectories are denoised jointly, each conditioned on its neighbors' overlapping segments, and then merged into one long goal-reaching plan. An MLP inverse-dynamics model turns planned states into actions. Pointmaze uses a PD controller instead. The codebase derives from `diffuser` / `decision-diffuser`.

## Environment

- Conda env **must be named `compdfu_ogb_release`**. `diffuser/datasets/d4rl.py` checks `CONDA_DEFAULT_ENV` (`'ogb' in env name` → `Is_OgB_Robot_Env`), and the planning script asserts it. A differently named env silently takes the wrong code paths or fails.
- Setup: `pip install -r conda_env/requirements.txt`, then `./conda_env/install_pre.sh` (gymnasium, gymnasium-robotics, torch 2.5.0), then `./conda_env/install_ogb.sh`. The last one installs a customized OGBench fork (`devinluo27/ogbench_cpdfu_release`), so edit the path inside it first.
- Upstream `pip install ogbench "mujoco==3.2.6"` also works; mujoco ≥3.3 has no py3.9 wheel. The fork's extra env helpers (`get_maze_unit`, `str_maze_spec`, markers, …) are added to upstream envs by `diffuser/datasets/ogb_dset/ogb_compat.py:patch_env`, which `ogb_load_env*` calls. The fork only differs from upstream in visuals and helpers, not in dynamics. The pre-collected eval path (`ev_protocol=None`) still needs the fork's `set_state_with_obs` etc.
- OGBench datasets are downloaded to `~/.ogbench/data/` (see `conda_env/ogb_dset_download.ipynb`).
- Rendering backend: `train_ogb_stgl_sml.py`, `train_og_invdyn.py` and `train_dd_ogb.py` set `MUJOCO_GL`/`PYOPENGL_PLATFORM` to `osmesa` unless the hostname is `bishop`; the plan scripts use `egl`. On a headless NVIDIA machine use `egl`, since OSMesa may not be installed.

## Commands

There is no test suite, linter, or build step. Every entry point is a shell script with a hard-coded `config=...` line. Uncomment the config you want. The scripts double as SLURM batch files (`#SBATCH` headers) and also run fine with `sh`/`bash`. Run them from the repo root: the scripts use relative paths and do `sys.path.append('./')`.

```bash
# Train CompDiffuser planner          ($1 = GPU idx)
sh ./diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.sh 0
# Train inverse dynamics model
sh ./diffuser/ogb_task/og_inv_dyn/train_og_invdyn.sh 0
# Evaluate / roll out planner         ($1 GPU, $2 num episodes 1-100, $3 seed)
sh ./diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.sh 0 10 0
# Decision Diffuser baseline
./diffuser/baselines/dd_ogb/train_dd_ogb.sh ; ./diffuser/baselines/dd_ogb/plan_dd_ogb.sh
# Full OGBench benchmark of one config  ($1 GPU, $2 n_seeds)
bash ./diffuser/ogb_task/ogb_maze_v1/run_ogb_benchmark.sh 0 8
```

`run_ogb_benchmark.py` (wrapped by the `.sh`) drives the whole protocol for one planner config: for each seed `0..N-1` it trains the planner, trains the inverse-dynamics model named by the config's `inv_model_path` (skipped when that is `None`, i.e. pointmaze), then evaluates the last three checkpoints and prints per-seed and across-seed success, also saved to `<logbase>/<dataset>/00_bench_<config>.json`. Training resumes from an existing checkpoint of the same seed by default (`--continue_training`, passes `--resume 1`), so the whole pipeline can just be relaunched after a crash or reboot; `--no-continue_training` restarts from scratch and `--skip_trained` skips training entirely when a checkpoint exists. Each of the three checkpoints is evaluated with its own rollout seed (`seed*100 + i`), so their start/goal problems differ as in OGBench. Useful flags: `--seeds 0,1,2`, `--n_ep_per_task` (50 = OGBench), `--n_last_ckpt`, `--dry_run`, `--train_extra`/`--plan_extra` (passthrough, e.g. `--train_extra "--n_train_steps 1000000 --n_saves 10"` to keep a checkpoint per 100K steps). It runs the seeds sequentially on one GPU; run several configs in parallel by launching one process per GPU. Note `n_saves` sets the checkpoint *label* spacing (`label_freq = n_train_steps // n_saves`). Every `save_freq` steps the trainer overwrites `state_<ceil(step / label_freq) * label_freq>.pt`, so once training passes step L, `state_L.pt` holds exactly the weights after L updates (keep `label_freq` a multiple of `save_freq`). The OGBench-protocol configs use `n_train_steps=1e6, n_saves=10`, so the last three checkpoints are 800K/900K/1M, as in OGBench. `plan_ogb_stgl_sml.py` evaluates `--diffusion_epoch <label>` when given, otherwise the latest checkpoint. In ogbench mode actions are clipped to [-1, 1] before `env.step`, as in `ogbench/impls/utils/evaluation.py` (the point env does not clip itself).

Or call the Python directly: `python diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py --config config/<...>.py`. Any extra `--key value` pair overrides a config field. The key must already exist in the config, except `plan_n_ep`, `diffusion_epoch`, and `config_2` (see `Parser.add_extras` in `diffuser/utils/setup.py`).

Run the scripts with `bash`, not `sh`; dash has no `source`.

The training scripts call `wandb.init(mode='online')` explicitly, which overrides the `WANDB_MODE` env var.

`--seed N` works on every entry point (declared on `utils.Parser`, so it is a real flag, not a config override). It seeds `random`/`numpy`/torch via `Parser.set_seed`, is recorded in `args.json`, and a non-zero seed is appended to the experiment name (`..._T512_sd1`), so seeds do not overwrite each other. Seed 0 is the default and keeps the original paths. Pass the same `--seed` at eval time, since the planner rebuilds the name to find the checkpoints. Runs are comparable but not bit-identical: the dataloader uses 6 workers with `shuffle=True` and no explicit generator, and `cudnn.benchmark` is on. The eval-time rollout seed is separate (`--pl_seeds`).

Quick smoke test: `config/og_antM_Gi_o2d_luotest.py`. Setting `dset_h5path` in a config's `base` dict (e.g. `data/ogb_maze/antmaze-giant-stitch-v0-luotest.npz`) swaps in a tiny dataset subset. It also disables wandb (`mode='disabled'` when `dset_h5path` is set, otherwise online logging to project `comp_diffuser_release`). Remove `dset_h5path` for real runs.

## Config system

- A config is a plain Python module under `config/` exposing `base = {'dataset': ..., 'diffusion': {...}, 'plan': {...}}`. `Parser.parse_args('diffusion' | 'plan')` loads the matching sub-dict onto `args`. A module-level dict named after the dataset (dashes → underscores) can override per-experiment keys.
- Class references in configs are **dotted strings relative to the `diffuser` package** (e.g. `'models.cd_stgl_sml_dfu.stgl_sml_diffusion_v1.Stgl_Sml_GauDiffusion_InvDyn_V1'`, `'datasets.ogb_dset.OgB_SeqDataset_V2'`). `utils.Config` resolves them via `import_class` and pickles its constructor kwargs to `<savepath>/*_config.pkl`. Loading at plan time rebuilds objects from these pickles, so renaming or moving a class breaks existing checkpoints.
- Save path is `logs/<dataset>/<exp_name>`. `exp_name` comes from `watch(diffusion_args_to_watch)`, which is essentially `diffusion/<config filename>`, so the config filename is the experiment identity. Plans go to `logs/<dataset>/plans/...`. `f:`-prefixed strings (e.g. `diffusion_loadpath`) are lazily formatted against `args`.
- Resume: `--resume 1` continues a run from the latest `state_*.pt` in its own savepath (crash/reboot recovery); the remaining epochs are `(n_train_steps - trainer.step) // n_steps_per_epoch`, and it prints `[resume] loaded state_<label>.pt, continue from step <n>`. It restores model/EMA/step but **not** the optimizer state, which checkpoints do not store, and it restarts from the last *saved* step, so up to `save_freq` steps are redone. Works for both `train_ogb_stgl_sml.py` and `train_og_invdyn.py`. The older config-driven form still works: a config named `<orig>_resume.py` with `trainer_dict['do_train_resume']=True` and `path_resume` containing `<orig>` in its name.
- Config filename conventions: `antM`/`pntM`/`humM`/`antSoc` = env, with a `nav`/`expl` suffix for navigate/explore datasets (none means stitch). `Gi/Lg/Me/Ar` = size. `o2d`/`o15d`/`o29d`/`o69d` = planned state dims (set via `dataset_config['obs_select_dim']`). `g*d` in invdyn configs = goal dims. `h160`/`ovlp56` = segment horizon / overlap length. `DiT...` = DiT backbone instead of the UNet.
- `max_path_length` must cover the episode length. V2 pads stitch/explore episodes (200 / 400 for humanoid / 500) by `extra_pad`, but navigate episodes (1000 / 2000 / 4000) are not padded. Inverse-dynamics configs need `max_path_length >= episode_len + horizon - 1`. With upstream OGBench, keep `sample_freq` > 0 only if `patch_env` is in place (training renders call the fork helpers).

## Architecture

- **Training pipeline** (`diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py`): dataset (`datasets/ogb_dset/OgB_SeqDataset_V2`, which pads episodes with a buffer via `pad_option_2='buf'`, `extra_pad`) → denoiser network → diffusion wrapper → trainer.
  - Denoiser: `models/cd_stgl_sml_dfu/stgl_sml_temporal_cond_v1.py` UNet1D, or `ogb_task/og_models/stgl_sml_dit_1d.py` DiT1D for high-dim planners. It takes the noisy segment plus encodings of the neighboring segments' overlap regions (`st_ovlp_model_config` / `end_ovlp_model_config`, encoders in `ogb_task/og_models/dit_1d_traj_encoder.py` and `models/cond_cp_dfu/`).
  - Diffusion wrapper: `models/cd_stgl_sml_dfu/stgl_sml_diffusion_v1.py`. During training it randomly drops one or both neighbor conditions or uses inpainting (`diff_config`: `tr_1side_drop_prob`, `tr_inpat_prob`, `tr_ovlp_prob`), so one model learns the start, middle, and end roles.
  - Trainer: `ogb_task/ogb_maze_v1/ogb_stgl_sml_training_v1.py`, on top of `models/cd_stgl_sml_dfu/stgl_sml_training_v1.py`. Checkpoints are saved as `state_<step>.pt`.
- **Compositional sampling**: `models/cd_stgl_sml_dfu/stgl_sml_policy_v1.py` (`Stgl_Sml_Policy_V1`) chains `ev_n_comp` segments. `args.ev_cp_infer_t_type` selects the scheme: `interleave` (default, autoregressive), `same_t` / `same_t_p` (parallel), `ar_back` (backward autoregressive), and `gsc` (Generative Skill Chaining baseline). Segments are merged with `guides/comp/traj_blender.py`.
- **Evaluation** (`ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py` → `OgB_Stgl_Sml_MazeEnvPlanner_V1` in `ogb_stgl_sml_planner_v1.py`) has two modes.
  - **`ev_protocol='ogbench'`** (set in the config's `plan` dict) follows the original OGBench benchmark.
    - It runs `env.reset(options=dict(task_id=k))` for 5 tasks × `plan_n_ep` episodes per task (default 50).
    - Episode length is the env TimeLimit (1000; humanoid 2000, or 4000 for giant). Success is taken at the final step, and the episode stops on `terminated`.
    - It writes `task{k}_success` and `overall_success` to `00_rollout.json`.
    - The eval hyper-parameters (`ev_n_comp`, `repl_ada_dist_cfg`, `inv_model_path`, `inv_epoch`, …) are read from the config.
  - **Legacy mode** (no `ev_protocol`) is how the paper was evaluated.
    - Per-environment eval hyper-parameters are **hard-coded in the `__main__` block of `plan_ogb_stgl_sml.py`**, in an `if dataset ...` chain.
    - The start/goal pairs are fixed (5 tasks × 20) from `data/ogb_maze/ev_probs/*.hdf5`.
    - Step budgets are raised above OGBench's (e.g. 2000–8000).
  - Both modes write `.mp4` and `.json` results. In ogbench mode, only the first episode of each task is recorded to video.
- **Env/path registries** (legacy mode): `diffuser/utils/ogb_utils/ogb_serial.py` maps:
  - env name → eval-problem file (`get_ogb_maze_ev_probs_fname`);
  - dataset normalizer constants (datasets not in the table fall back to the loaded training dataset's normalizer, which requires full-obs planners);
  - env name + goal dim → a **hard-coded inverse-dynamics checkpoint dir under `logs/`** (`ogb_get_inv_model_path`). ogbench-mode configs bypass this with `inv_model_path`.
- **Inverse dynamics**: `diffuser/ogb_task/og_inv_dyn/` (MLP: current full state + planned goal state → action). Configs are in `config/ogb_invdyn/`.
- Other code: `diffuser/baselines/` (Decision Diffuser for OGB and Maze2D), `diffuser/pl_eval/gen_ev_probs/` (eval-problem generation), `diffuser/utils/ogb_paper_vis_utils/` (multi-agent MuJoCo rendering for paper figures). `data/m2d` and the D4RL-related code are legacy Maze2D support.

## Pretrained models

Download the checkpoint zips (links in README.md) and unzip them at the repo root. They extract into `logs/<env>/diffusion/<exp_name>/`. Antmaze also needs the matching inverse-dynamics zip.
