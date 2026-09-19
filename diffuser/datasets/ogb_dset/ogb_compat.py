import types
import numpy as np

## ---------- Compatibility with upstream OGBench ----------
## This codebase was written against a customized OGBench fork (ogbench_cpdfu_release),
## which adds helper methods to the maze env. Those helpers only read/write state or
## move visual markers, so we attach equivalent versions to an upstream env instance.
## Methods that already exist (i.e., the fork is installed) are left untouched.


def get_str_maze_spec_from_map(maze_map):
    '''maze_map: np 2d, 1 is wall. Returns a d4rl-style string, rows joined by "\\"'''
    rows = [''.join('#' if c == 1 else 'O' for c in row) for row in maze_map]
    return '\\'.join(rows)


def _get_maze_unit(self):
    return self._maze_unit

def _get_offset_x(self):
    return self._offset_x

def _get_offset_y(self):
    return self._offset_y

def _get_qpos_qvel(self):
    return np.concatenate([self.data.qpos, self.data.qvel])

def _set_seed_addn(self, seed_addn):
    ## the fork seeds start/goal noise with its own rng; upstream uses global np.random
    pass

def _set_marker_if_exists(geom_name):
    def _fn(self, xy):
        try:
            geom = self.model.geom(geom_name)
        except KeyError:
            return ## upstream env has no such visual marker
        geom.pos[:2] = xy
    return _fn


def patch_env(env):
    '''env: an unwrapped ogbench maze env, patched in place and returned'''
    new_methods = dict(
        get_maze_unit=_get_maze_unit,
        get_offset_x=_get_offset_x,
        get_offset_y=_get_offset_y,
        get_qpos_qvel=_get_qpos_qvel,
        set_seed_addn=_set_seed_addn,
        set_subgoal_waypnt=_set_marker_if_exists('subgoal_waypnt'),
        set_start_marker=_set_marker_if_exists('start_marker'),
        set_ball_start_marker=_set_marker_if_exists('ball_start_marker'),
    )
    for name, fn in new_methods.items():
        if not hasattr(env, name):
            setattr(env, name, types.MethodType(fn, env))

    if not hasattr(env, 'str_maze_spec') and hasattr(env, 'maze_map'):
        env.str_maze_spec = get_str_maze_spec_from_map(env.maze_map)

    return env
