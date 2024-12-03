from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class InAirReward(RewardFunction): 
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if not player.on_ground:
            return 1
        else:
            return 0