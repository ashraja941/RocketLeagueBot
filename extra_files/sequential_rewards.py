from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from numpy import ndarray

class SequentialRewards(RewardFunction):
    def __init__(self, rewards: list, steps: list):
        super().__init__()
        self.rewards_list = rewards
        self.step_counts = steps
        self.step_count = 0
        self.step_index = 0
        assert len(self.rewards_list) == len(self.step_counts)

    def reset(self, initial_state: GameState):
        for rew in self.rewards_list:
            rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        if self.step_index < len(self.step_counts) and self.step_count > self.step_counts[self.step_index]:
            self.step_index += 1

        self.step_count += 1
        return self.rewards_list[self.step_index].get_reward(player, state, previous_action)
