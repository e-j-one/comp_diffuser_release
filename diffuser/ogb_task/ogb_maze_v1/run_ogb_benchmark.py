"""
Run the full OGBench benchmark pipeline of one config, for several training seeds.

For each seed:
    1. train the CompDiffuser planner
    2. train the inverse dynamics model (skipped for pointmaze, which uses a PD controller)
    3. evaluate the last three checkpoints, following the OGBench protocol
       (5 tasks x 50 episodes, env TimeLimit, final-step success)

The reported number of a seed is the average over its last three checkpoints, as in
OGBench (Appendix E: 'the average success rate across the last three evaluation epochs').

Example:
    python diffuser/ogb_task/ogb_maze_v1/run_ogb_benchmark.py \
        --config config/ogb_ant_maze_nav/og_antMnav_Me_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py \
        --n_seeds 8 --gpu 0
"""
import argparse, glob, importlib.util, json, os, subprocess, sys
import os.path as osp

sys.path.append('./')

TRAIN_PY = 'diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py'
INV_PY = 'diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py'
PLAN_PY = 'diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py'


def load_config_base(config_path):
    '''import a config module by file path and return its `base` dict'''
    spec = importlib.util.spec_from_file_location('cfg_mod', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.base


def find_inv_config(inv_model_path):
    '''inv_model_path is logs/<dataset>/diffusion/<inv cfg name>, find that config file'''
    inv_name = osp.basename(inv_model_path.rstrip('/'))
    hits = glob.glob(f'config/ogb_invdyn/**/{inv_name}.py', recursive=True)
    assert len(hits) == 1, f'expect 1 config for {inv_name}, found {hits}'
    return hits[0]


def exp_name_of(config_path, base, seed):
    '''
    same rule as utils/setup.py watch(): <config_fn>[_T<n_diffusion_steps>][_sd<seed>].
    the inv dyn configs have no 'n_diffusion_steps', so watch() skips that part
    '''
    name = osp.splitext(osp.basename(config_path))[0]
    n_dfu_steps = base['diffusion'].get('n_diffusion_steps', None)
    if n_dfu_steps is not None:
        name = f'{name}_T{n_dfu_steps}'
    return f'{name}_sd{seed}' if seed else name


def ckpt_labels(logdir):
    '''the state_<label>.pt labels present, sorted ascending'''
    labels = [int(osp.basename(p)[len('state_'):-len('.pt')])
              for p in glob.glob(osp.join(logdir, 'state_*.pt'))]
    return sorted(labels)


def run(cmd, env, dry_run):
    print('\n[ run_ogb_benchmark ] $ ' + ' '.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env)


def newest_result_json(logbase, dataset, seed):
    '''the 00_rollout.json of the most recent ogbench-protocol eval of this run'''
    pat = osp.join(logbase, dataset, 'plans', '*', '*', '*-ogbEv*', '00_rollout.json')
    cands = glob.glob(pat)
    if seed: ## the plan exp_name also carries the seed suffix
        cands = [c for c in cands if f'_sd{seed}' in c]
    else:
        cands = [c for c in cands if '_sd' not in c]
    return max(cands, key=osp.getmtime) if cands else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='planner config, e.g. config/ogb_ant_maze_nav/....py')
    p.add_argument('--n_seeds', type=int, default=8, help='train seeds 0..N-1')
    p.add_argument('--seeds', type=str, default=None, help='explicit seed list, e.g. "0,1,2" (overrides --n_seeds)')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--n_ep_per_task', type=int, default=50, help='eval episodes per task, ogbench uses 50')
    p.add_argument('--n_last_ckpt', type=int, default=3, help='how many of the last checkpoints to evaluate')
    p.add_argument('--logbase', type=str, default='logs')
    p.add_argument('--train_extra', type=str, default='', help='extra args for both training scripts, e.g. "--n_train_steps 1000000 --n_saves 10"')
    p.add_argument('--plan_extra', type=str, default='', help='extra args for the eval script')
    p.add_argument('--skip_trained', action='store_true', help='skip training if checkpoints already exist')
    p.add_argument('--dry_run', action='store_true', help='only print the commands')
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(range(args.n_seeds))
    base = load_config_base(args.config)
    dataset = base['dataset']
    inv_model_path = base['plan'].get('inv_model_path', None)
    assert base['plan'].get('ev_protocol') == 'ogbench', \
        f"{args.config} has no 'ev_protocol': 'ogbench' in its plan dict"
    inv_config = find_inv_config(inv_model_path) if inv_model_path else None

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    ## egl renders on the GPU without a display; do NOT set MUJOCO_EGL_DEVICE_ID,
    ## mujoco then picks a working egl device itself
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('PYOPENGL_PLATFORM', env['MUJOCO_GL'])

    print(f'[ run_ogb_benchmark ] {dataset=} {seeds=} inv_config={inv_config}', flush=True)

    results = {} ## seed -> {label: overall_success}
    for seed in seeds:
        seed_args = ['--seed', str(seed), '--logbase', args.logbase]
        logdir = osp.join(args.logbase, dataset, 'diffusion', exp_name_of(args.config, base, seed))

        ## ---------------- 1. planner ----------------
        if args.skip_trained and ckpt_labels(logdir):
            print(f'[ run_ogb_benchmark ] skip planner training, found ckpt in {logdir}', flush=True)
        else:
            cmd = [sys.executable, TRAIN_PY, '--config', args.config] + seed_args + args.train_extra.split()
            if run(cmd, env, args.dry_run) != 0:
                print(f'[ run_ogb_benchmark ] planner training FAILED, {seed=}', flush=True)
                continue

        ## ---------------- 2. inverse dynamics ----------------
        if inv_config is not None:
            inv_base = load_config_base(inv_config)
            inv_logdir = osp.join(args.logbase, dataset, 'diffusion',
                                  exp_name_of(inv_config, inv_base, seed))
            if args.skip_trained and ckpt_labels(inv_logdir):
                print(f'[ run_ogb_benchmark ] skip inv dyn training, found ckpt in {inv_logdir}', flush=True)
            else:
                cmd = [sys.executable, INV_PY, '--config', inv_config] + seed_args + args.train_extra.split()
                if run(cmd, env, args.dry_run) != 0:
                    print(f'[ run_ogb_benchmark ] inv dyn training FAILED, {seed=}', flush=True)
                    continue

        ## ---------------- 3. eval the last three checkpoints ----------------
        labels = ckpt_labels(logdir)[-args.n_last_ckpt:]
        if not labels and args.dry_run:
            labels = ['<label of each of the last 3 ckpts>'] ## nothing trained yet
        if not labels:
            print(f'[ run_ogb_benchmark ] no checkpoint in {logdir}, skip eval', flush=True)
            continue
        print(f'[ run_ogb_benchmark ] {seed=} eval checkpoints: {labels}', flush=True)

        results[seed] = {}
        for label in labels:
            cmd = [sys.executable, PLAN_PY, '--config', args.config,
                   '--plan_n_ep', str(args.n_ep_per_task), '--pl_seeds', str(seed),
                   '--diffusion_epoch', str(label)] + seed_args + args.plan_extra.split()
            if run(cmd, env, args.dry_run) != 0:
                print(f'[ run_ogb_benchmark ] eval FAILED, {seed=} {label=}', flush=True)
                continue
            if args.dry_run:
                continue
            j_path = newest_result_json(args.logbase, dataset, seed)
            if j_path is None:
                print(f'[ run_ogb_benchmark ] no result json found, {seed=} {label=}', flush=True)
                continue
            with open(j_path) as f:
                j_data = json.load(f)
            results[seed][label] = j_data['overall_success']
            print(f'[ run_ogb_benchmark ] {seed=} {label=} '
                  f"overall_success={j_data['overall_success']:.4f}", flush=True)

    ## ---------------- summary ----------------
    print('\n' + '=' * 70)
    print(f'[ run_ogb_benchmark ] {dataset} | {args.config}')
    print(f'ogbench protocol: 5 tasks x {args.n_ep_per_task} episodes, '
          f'avg over the last {args.n_last_ckpt} checkpoints')
    seed_means = {}
    for seed, per_ckpt in sorted(results.items()):
        if not per_ckpt:
            continue
        seed_means[seed] = sum(per_ckpt.values()) / len(per_ckpt)
        detail = ', '.join(f'{k}: {v * 100:.1f}' for k, v in sorted(per_ckpt.items()))
        print(f'  seed {seed}: {seed_means[seed] * 100:.1f}%  ({detail})')
    if seed_means:
        vals = list(seed_means.values())
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f'  ---> {mean * 100:.1f}% +- {std * 100:.1f}% over {len(vals)} seeds')
    out_path = osp.join(args.logbase, dataset,
                        f'00_bench_{osp.splitext(osp.basename(args.config))[0]}.json')
    if not args.dry_run and results:
        with open(out_path, 'w') as f:
            json.dump(dict(config=args.config, dataset=dataset,
                           n_ep_per_task=args.n_ep_per_task,
                           per_seed_per_ckpt=results, per_seed=seed_means), f, indent=2)
        print(f'  saved: {out_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
