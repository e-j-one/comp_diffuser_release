#!/bin/bash

#SBATCH --job-name=script-ev-hi
#SBATCH --output=trash/slurm/plan_OG_StglSml_Jan27/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
### brainiac, 2080: alexa,alexa
#SBATCH --exclude="clippy,voltron,claptrap,alexa,bmo,olivaw,oppy"

##SBATCH --partition="rl2-lab"

#SBATCH --gres=gpu:a40:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --qos="short"

##SBATCH --gres=gpu:rtx_6000:1
##SBATCH --gres=gpu:a5000:1

#SBATCH --qos="debug"
#SBATCH --time=16:00:00

echo $(hostname)

source ~/.bashrc
source activate compdfu_ogb_release

# config=""

# config="config/og_antM_Gi_o2d_luotest.py" ## test only, placeholder config file


## --------- Planning: OGBench AntMaze Stitch 15/29D Planner ----------
## i.e., the diffusion planner generates trajectory of shape (Horizon, 15/29)
# config="config/ogb_ant_maze/og_antM_Lg_o15d_PadBuf_Ft64_ts512_DiTd768dp16_fs4_h160_ovlp56Mdit.py"
# config="config/ogb_ant_maze/og_antM_Lg_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"


## --------- Planning: OGBench AntMaze Stitch 2D Planner ----------
## Giant
# config="config/ogb_ant_maze/og_antM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Large
# config="config/ogb_ant_maze/og_antM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Medium
# config="config/ogb_ant_maze/og_antM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"

## --------- Planning: OGBench AntMaze Explore 2D Planner ----------
## TODO:


## --------- Planning: OGBench PointMaze Stitch 2D Planner ----------
## Giant
config="config/ogb_pnt_maze/og_pntM_Gi_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Large
# config="config/ogb_pnt_maze/og_pntM_Lg_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"
## Medium
# config="config/ogb_pnt_maze/og_pntM_Me_o2d_Cd_Stgl_PadBuf_Ft64_ts512.py"




## ------ High-dim planners on navigate / explore / stitch (OGBench-protocol eval) ------
## AntMaze Stitch 29D
# config="config/ogb_ant_maze/og_antM_Me_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze/og_antM_Gi_o29d_DiTd1024dp12_PadBuf_Ft64_fs4_h160_ovlp56MditD512.py"
## AntMaze Navigate 29D
# config="config/ogb_ant_maze_nav/og_antMnav_Me_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze_nav/og_antMnav_Lg_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"
# config="config/ogb_ant_maze_nav/og_antMnav_Gi_o29d_DiTd1024dp12_fs4_h160_ovlp56MditD512.py"
## AntMaze Explore 29D
# config="config/ogb_ant_maze_expl/og_antMexpl_Me_o29d_DiTd1024dp12_fs6_h192_ovlp66MditD512.py"
# config="config/ogb_ant_maze_expl/og_antMexpl_Lg_o29d_DiTd1024dp12_fs6_h192_ovlp66MditD512.py"
## HumanoidMaze Stitch 69D
# config="config/ogb_hum_maze/og_humM_Me_o69d_DiTd1024dp12_PadBuf_Ft64_fs4_h336_ovlp128MditD512.py"
# config="config/ogb_hum_maze/og_humM_Lg_o69d_DiTd1024dp12_PadBuf_Ft64_fs4_h336_ovlp128MditD512.py"
# config="config/ogb_hum_maze/og_humM_Gi_o69d_DiTd1024dp12_PadBuf_Ft64_fs4_h336_ovlp128MditD512.py"
## HumanoidMaze Navigate 69D
# config="config/ogb_hum_maze_nav/og_humMnav_Me_o69d_DiTd1024dp12_fs4_h336_ovlp128MditD512.py"
# config="config/ogb_hum_maze_nav/og_humMnav_Lg_o69d_DiTd1024dp12_fs4_h336_ovlp128MditD512.py"
# config="config/ogb_hum_maze_nav/og_humMnav_Gi_o69d_DiTd1024dp12_fs4_h336_ovlp128MditD512.py"
## PointMaze Navigate 2D
# config="config/ogb_pnt_maze_nav/og_pntMnav_Me_o2d_Cd_Stgl_Ft64_ts512.py"
# config="config/ogb_pnt_maze_nav/og_pntMnav_Lg_o2d_Cd_Stgl_Ft64_ts512.py"
# config="config/ogb_pnt_maze_nav/og_pntMnav_Gi_o2d_Cd_Stgl_Ft64_ts512.py"


{

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
MUJOCO_EGL_DEVICE_ID=${1:-0} \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config $config \
    --plan_n_ep $2 \
    --pl_seeds ${3:--1} \


exit 0

}