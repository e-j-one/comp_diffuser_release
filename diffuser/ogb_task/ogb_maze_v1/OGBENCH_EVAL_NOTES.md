# CompDiffuser under the OGBench protocol: settings and disclosures

Notes for reporting numbers produced by `run_ogb_benchmark.py` next to the OGBench paper tables.

## Protocol (matches OGBench)

- 1M training steps; checkpoints at 800K / 900K / 1M; the reported number per seed is the mean over those three.
- 8 training seeds; each checkpoint is evaluated with its own rollout seed (`seed*100 + i`).
- `env.reset(options=dict(task_id=k))` for 5 tasks x 50 episodes; goal = `info['goal']` (full goal observation).
- Episode length = registered TimeLimit (1000; humanoidmaze 2000, humanoidmaze-giant 4000); episode ends on `terminated`; success = `info['success']` at the final step; overall = mean of the 5 per-task rates.
- Actions clipped to [-1, 1] before `env.step`.
- Trained only on the train split of the benchmark dataset (`~/.ogbench/data/<dataset>.npz`); the inverse-dynamics model uses the same dataset.

## Eval hyper-parameters (per config, `plan` dict)

Taken from the CompDiffuser paper (its hard-coded table in `plan_ogb_stgl_sml.py`). Navigate datasets reuse the
stitch settings of the same maze; they were not tuned separately.

Common to all: `ev_cp_infer_t_type='interleave'`, `n_act_per_waypnt=1`, `is_replan='ada_dist'`, replan `type='m_2'`,
`cond_2_extra=150`, 40 samples per plan, DDIM 50 steps (eta 1.0), guidance weight 2.0, top-5 by overlap consistency, pick first.

| Dataset | `ev_n_comp` | `thres` | `max_n_repl` | `ada_dist_minus_n_wp` | replan distance on |
|---|---|---|---|---|---|
| antmaze-medium-stitch / -navigate | 3 | 4 | 10 | 50 | xy |
| antmaze-large-stitch / -navigate | 5 | 4 | 10 | 50 | xy |
| antmaze-giant-stitch / -navigate | 9 | 4 | 15 | 0 | xy |
| antmaze-medium-explore | 5 | 2 | 10 | 0 | xy |
| antmaze-large-explore | 10 | 2 | 10 | 0 | xy |
| humanoidmaze-medium-stitch / -navigate | 4 | 10 | 10 | 300 | xy |
| humanoidmaze-large-stitch / -navigate | 6 | 10 | 10 | 300 | xy |
| humanoidmaze-giant-stitch / -navigate | 11 | 10 | 10 | 300 | xy |
| pointmaze-medium-stitch / -navigate | 3 | 1 | 0 (no replanning) | 0 | xy |
| pointmaze-large-stitch / -navigate | 6 | 1 | 0 (no replanning) | 0 | xy |
| pointmaze-giant-stitch / -navigate | 8 | 1 | 10 | 10 | xy |

- `ev_n_comp`: number of short segments composed into one plan (scales with maze size).
- `thres`: replan when the agent is farther than this from its current waypoint.
- `max_n_repl`: maximum replans per episode.
- `ada_dist_minus_n_wp`: shortens the replanned horizon relative to the remaining plan.

## Deviations to disclose

1. **Per-dataset eval hyper-parameters** (table above). They were chosen per maze size and robot, and likely on the
   paper's fixed eval problems (`data/ogb_maze/ev_probs/*.hdf5`), which appear to be samples of the same 5 OGBench tasks.
   To be strict, fix them per environment family or re-select them on held-out start/goal pairs.
2. **Pointmaze uses a hand-coded PD controller** (`action = 5 * (waypoint - obs)`, then clipped), which inverts the
   env's known `qpos += 0.2 * action` rule, instead of a learned policy or inverse-dynamics model.
3. **Batch size 128** for the planner (OGBench default: 1024); inverse dynamics uses 1024. The planner and the
   inverse-dynamics model are each trained for 1M steps.
4. **Test-time compute**: 40 samples x 50 DDIM steps per plan, up to 10-15 replans per episode; large DiT backbone
   (width 1024, 12 layers) for ant/humanoid.
5. **Resume restarts the optimizer**: checkpoints store model/EMA/step only; a resumed run starts AdamW from fresh moments.
