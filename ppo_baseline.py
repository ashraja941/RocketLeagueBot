import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import TouchBallReward,VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward,FaceBallReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import DiscreteAction
    from rlgym_sim.utils.state_setters import RandomState
    from extra_files.air_rewards import InAirReward
    from extra_files.sequential_rewards import SequentialRewards

    import rocketsimvis_rlgym_sim_client as rsv
    
    spawn_opponents = False
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine_1 = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          FaceBallReward(),
                          EventReward(team_goal=1, concede=-1, demo=0.1),
                          TouchBallReward(),
                          InAirReward()
                          )
    reward_weights_1 = (3,1, 1, 10.0,20,1)

    reward_fn_1 = CombinedReward(reward_functions=rewards_to_combine_1,
                               reward_weights=reward_weights_1)
    
    rewards_to_combine_2 = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          FaceBallReward(),
                          EventReward(team_goal=1, concede=-1, demo=0.1),
                          TouchBallReward(),
                          InAirReward()
                          )
    reward_weights_2 = (3,8, 1, 20.0,1,0)

    reward_fn_2 = CombinedReward(reward_functions=rewards_to_combine_2,
                               reward_weights=reward_weights_2)
    
    rewards_order = [reward_fn_1,reward_fn_2]
    step_requirements = [25000000,70000000]
    reward_fn = SequentialRewards(rewards_order,step_requirements)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    state_setter = RandomState(False,False,True)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      render=True,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=70_000_000,
                      log_to_wandb=True)
    learner.learn()