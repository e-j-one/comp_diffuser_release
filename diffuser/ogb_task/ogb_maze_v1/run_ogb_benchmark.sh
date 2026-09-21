#!/bin/bash

## Full OGBench benchmark of one config: for each training seed, train the planner,
## train the inverse dynamics model (except pointmaze), then evaluate the last three
## checkpoints (5 tasks x 50 episodes each).
## usage: bash ./diffuser/ogb_task/ogb_maze_v1/run_ogb_benchmark.sh $gpu $n_seeds
## training continues from an existing checkpoint of the same seed by default,
## so this can simply be relaunched after a crash or a reboot
## e.g.:  bash ./diffuser/ogb_task/ogb_maze_v1/run_ogb_benchmark.sh 0 8

source ~/.bashrc
source activate compdfu_ogb_release

## optionally 'cd' to 'comp_diffuser_release' folder
# cd $Your_Folder_of_This_Repo

## --------- AntMaze Stitch 29D ----------
# config="config/ogb_ant_maze/og_antM_Me_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze/og_antM_Lg_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze/og_antM_Gi_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"

## --------- AntMaze Navigate 29D ----------
config="config/ogb_ant_maze_nav/og_antMnav_Me_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze_nav/og_antMnav_Lg_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze_nav/og_antMnav_Gi_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"

## --------- AntMaze Explore 29D ----------
# config="config/ogb_ant_maze_expl/og_antMexpl_Me_o29d_DiTd1024dp12_fs6_h192_ovlp66MditD512.py"
# config="config/ogb_ant_maze_expl/og_antMexpl_Lg_o29d_DiTd1024dp12_fs6_h192_ovlp66MditD512.py"

## --------- HumanoidMaze Stitch / Navigate 69D ----------
# config="config/ogb_hum_maze/og_humM_Me_o69d_DiTd1024dp12_PadBuf_Ft64_fs4_h336_ovlp128MditD512.py"
# config="config/ogb_hum_maze_nav/og_humMnav_Me_o69d_DiTd1024dp12_fs4_h336_ovlp128MditD512.py"

## --------- PointMaze Stitch / Navigate 2D (no inverse dynamics model) ----------
# config="config/ogb_pnt_maze/og_pntM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
# config="config/ogb_pnt_maze_nav/og_pntMnav_Me_o2d_Cd_Stgl_Ft64_ts512.py"

## ogbench trains for 1M steps and evaluates every 100K; keeping a checkpoint per
## 100K steps needs n_saves = n_train_steps / 100000
train_extra="--n_train_steps 1000000 --n_saves 10"

{

echo $(hostname)

python diffuser/ogb_task/ogb_maze_v1/run_ogb_benchmark.py \
    --config $config \
    --gpu ${1:-0} \
    --n_seeds ${2:-8} \
    --train_extra "$train_extra" \

exit 0

}
