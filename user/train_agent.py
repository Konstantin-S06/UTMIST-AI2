'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below.

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.01,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space = 64, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward /140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0


def smart_attack_reward(env: WarehouseBrawl) -> float:
    """Reward attacking when the opponent is near, punish otherweise."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = math.sqrt(dx**2 + dy**2)
    is_attacking = isinstance(player.state, AttackState)

    if is_attacking and distance <= 0.5:
        if opponent.damage_taken_this_frame > 0:
            return 4.0 * env.dt + opponent.damage_taken_this_frame * 0.5 * env.dt
        else:
            return -2 * env.dt
    elif is_attacking and distance > 1:
        return -4.0 * env.dt
    return 0.0



def using_correct_attack_reward(env: WarehouseBrawl) -> float:
    """Reward using attacks appropriate for the opponent's relative position."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    if not isinstance(player.state, AttackState) or abs(player.body.position.x - opponent.body.position.x) > 1.0:
        return 0.0

    # Determine the relative position of the opponent
    dx = opponent.body.position.x - player.body.position.x
    dy = opponent.body.position.y - player.body.position.y
    x = player.body.position.x

    # Determine the intended attack direction based on input
    keys = player.input.key_status
    up    = keys["W"].held
    down  = keys["S"].held
    left  = keys["A"].held
    right = keys["D"].held

    if up and right:
        attack_direction = "diagonal_up_right"
    elif up and left:
        attack_direction = "diagonal_up_left"
    elif down and right:
        attack_direction = "diagonal_down_right"
    elif down and left:
        attack_direction = "diagonal_down_left"
    elif up:
        attack_direction = "up"
    elif down:
        attack_direction = "down"
    elif right or left:
        attack_direction = "horizontal"
    else:
        attack_direction = "neutral"

    # Check if attack matches opponent's relative position

    if dy < -0.1 and abs(dx) > 0.5:
        if attack_direction in ["diagonal_up_right", "diagonal_up_left", "neutral"] and opponent.damage_taken_this_frame > 0:
            if abs(x) < 6 and abs(x) > 3:
                return 5.5 * env.dt
        elif attack_direction in ["diagonal_up_right", "diagonal_up_left", "neutral"] and opponent.damage_taken_this_frame == 0:
            return 1.0 * env.dt
    elif dy < -0.1 and abs(dx) <= 0.5:  # opponent above
        if attack_direction in ["up", "neutral"] and opponent.damage_taken_this_frame > 0:
            if abs(x) < 7 and abs(x) > 2:
                return 5.5 * env.dt
        elif attack_direction in ["up", "neutral"] and opponent.damage_taken_this_frame == 0:
            return 1.0 * env.dt
    elif dy > 0.1 and abs(dx) > 0.5:  
        if attack_direction in ["diagonal_down_right", "diagonal_down_left"] and opponent.damage_taken_this_frame > 0:
            if abs(x) < 6 and abs(x) > 3:
                return 5.5 * env.dt
        elif attack_direction in ["diagonal_down_right", "diagonal_down_left"] and opponent.damage_taken_this_frame == 0:
            return 1.0 * env.dt
    elif dy > 0.1 and abs(dx) <= 0.5:  
        if attack_direction in ["down"] and opponent.damage_taken_this_frame > 0:
            if abs(x) < 7 and abs(x) > 2:
                return 5.5 * env.dt
        elif attack_direction in ["down"] and opponent.damage_taken_this_frame == 0:
            return env.dt
    else:  # opponent roughly at same height
        if attack_direction in ["horizontal", "neutral"] and opponent.damage_taken_this_frame > 0:
            if abs(x) < 6 and abs(x) > 3 or player.body.velocity.y == 0:
                return 5.5 * env.dt
        elif attack_direction in ["horizontal"] and opponent.damage_taken_this_frame == 0:
            return env.dt

    return -1.0 * env.dt

def face_right_direction_reward(env: WarehouseBrawl) -> float:
    """Punish facing away from the opponent."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    dx = opponent.body.position.x - player.body.position.x

    if dx >= 0 and player.facing == Facing.RIGHT:
        return 1.5 * env.dt
    elif dx < 0 and player.facing == Facing.LEFT:
        return 1.5 * env.dt
    return -0.5 *env.dt

def jump_recovery_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]

    x = player.body.position.x

    y_pos = player.body.position.y
    y_vel = player.body.velocity.y

    if x < 0 and 2.85 < y_pos:
        if y_vel > 0:
            return -10.0 * env.dt
    elif x > 0 and 7.0 < y_pos:
        if y_vel > 0:
            return -10.0 * env.dt
    return 0.0

def return_to_platform_reward(env: WarehouseBrawl) -> float:
    """Reward staying above or on the main platforms."""
    player: Player = env.objects["player"]
    x = player.body.position.x
    if (-7.0 < x < -2.0) or (2.0 < x < 7.0) or (player.body.velocity.y <= 0 and abs(x) < 2.0) and player.body.position.y > -4.0:
        return env.dt
    elif player.damage_taken_this_stock > 50:
        return -5.0 * env.dt
    elif abs(x) >= 7.0:
        return -3.0 * env.dt
    else:
        return -2.0 * env.dt
    
def return_to_platform_gentle(env: WarehouseBrawl) -> float:
    """Gentle encouragement to stay on platforms."""
    player: Player = env.objects["player"]
    x = player.body.position.x
    y = player.body.position.y
    
    # Small reward for being on main platforms
    if (-7.0 < x < -2.0 and y < 2.5  and y > -4.0) or (2.0 < x < 7.0 and y < 2.5 and y > -4.0):
        return 1.0 * env.dt
    else:
        return -5.0 * env.dt
    
    
    # NO PENALTY for normal off-platform play!
    return 0.0

def spike_reward(env: WarehouseBrawl) -> float:
    """Rewards spiking opponents when theyre off the platform."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    xp = player.body.position.x
    yp = player.body.position.y
    xo = opponent.body.position.x
    yo= opponent.body.position.y

    # If our player is directly above opponent
    if (abs(xp-xo) < 1) and (yp < yo):
        # If using ground pound
        if (isinstance(player.state, AttackState) and player.state.move_type in [MoveType.DAIR, MoveType.DSIG, MoveType.DLIGHT]):
                # If we still have a jump
                if isinstance(player.state, InAirState) and player.state.jumps_left > 0:
                    # If opponent takes damage this frame
                    if opponent.damage_taken_this_frame > 0:
                        # If opponent is off the platform
                        if abs(xo) > 7.0 and abs(xo) < 2.0:
                            return 22.0 * env.dt
                        return 15.0 * env.dt
                    else:
                        return -4.0 * env.dt
                return -20.0 * env.dt
    return 0.0

def dodge_attack_reward(env: WarehouseBrawl) -> float:
    """Rewards dodging when the opponent is attacking."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = math.sqrt(dx**2 + dy**2)
    if isinstance(player.state, DodgeState) and distance < 0.6:
        if isinstance(opponent.state, AttackState):
            if isinstance(player.state, InAirState):
                return 1.5 * env.dt
            return 4.0 * env.dt
        return -1.0 * env.dt
    return 0.0

def combo_reward(env: WarehouseBrawl) -> float:
    """
    Rewards landing consecutive hits in quick succession (combos).
    Tracks hits within a time window and rewards combo extensions.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Initialize combo tracking on player
    if not hasattr(player, 'combo_hits'):
        player.combo_hits = 0
        player.combo_damage = 0.0
        player.last_hit_frame = -100

    reward = 0.0
    current_frame = env.steps
    frames_since_last_hit = current_frame - player.last_hit_frame
    
    # Combo window: 30 frames (~1 second at 30fps)
    COMBO_WINDOW = 30
    
    # Check if opponent just got hit
    if opponent.damage_taken_this_frame > 0:
        # Within combo window - extend combo
        if frames_since_last_hit < COMBO_WINDOW:
            player.combo_hits += 1
            player.combo_damage += opponent.damage_taken_this_frame
            
            # Reward scales with combo length
            if player.combo_hits == 2:
                reward = 5.0  # Started a combo
            elif player.combo_hits == 3:
                reward = 10.0  # 3-hit combo
            elif player.combo_hits == 4:
                reward = 15.0  # 4-hit combo
            elif player.combo_hits >= 5:
                reward = 25.0 + (player.combo_hits - 5) * 10.0  # 5+ hit combo
        else:
            # Reset combo - new hit chain
            player.combo_hits = 1
            player.combo_damage = opponent.damage_taken_this_frame
        
        player.last_hit_frame = current_frame
    
    # Reset combo if window expired
    elif frames_since_last_hit > COMBO_WINDOW and player.combo_hits > 0:
        player.combo_hits = 0
        player.combo_damage = 0.0
    
    return reward * env.dt

def recovery_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]

    y_pos = player.body.position.y
    y_vel = player.body.velocity.y

    if 2.85 < y_pos:
        if y_vel > 0:
            return -30.0 * env.dt

    return 0.0

def string_reward(env: WarehouseBrawl) -> float:
    """
    Rewards hitting stunned opponents (true combos/strings).
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # If we hit opponent while they're in stun (can't escape)
    if opponent.damage_taken_this_frame > 0 and isinstance(opponent.state, StunState):
        return 1 * env.dt
    
    return 0.0

def combo_finisher_reward(env: WarehouseBrawl) -> float:
    """
    Rewards finishing combos with heavy/signature moves.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Must have active combo
    if not hasattr(player, 'combo_hits') or player.combo_hits < 2:
        return 0.0
    
    reward = 0.0
    
    # Check if using a finisher move
    if isinstance(player.state, AttackState) and hasattr(player.state, 'move_type'):
        move_type = player.state.move_type
        
        # Heavy attacks as finishers
        if move_type in [MoveType.NSIG, MoveType.DSIG, MoveType.SSIG]:
            reward = 10.0 * player.combo_hits
        
        # Spike finisher
        elif move_type == MoveType.GROUNDPOUND:
            reward = 15.0 * player.combo_hits
        
        # Recovery finisher (off-stage)
        elif move_type == MoveType.RECOVERY and abs(opponent.body.position.x) > 6.0:
            reward = 12.0 * player.combo_hits
    
    return reward * env.dt


def punish_combo_drop(env: WarehouseBrawl) -> float:
    """
    Punishes dropping combos (not following up after hitting).
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Must have started a combo
    if not hasattr(player, 'combo_hits') or player.combo_hits < 2:
        return 0.0
    
    current_frame = env.steps
    frames_since_last_hit = current_frame - player.last_hit_frame
    
    # If opponent recovered from stun and we didn't follow up
    if 15 < frames_since_last_hit < 30:
        if isinstance(opponent.state, (StandingState, WalkingState, InAirState)):
            if not isinstance(player.state, AttackState):
                # We had a chance to continue but didn't
                return -8.0 * env.dt
    
    return 0.0

def approach_opponent_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    xo = opponent.body.position.x
    xp = player.body.position.x
    yo = opponent.body.position.y
    yp = player.body.position.y

    keys = player.input.key_status
    jump = keys["space"].held

    if ((yo >= yp and not isinstance(player.state, InAirState)) or isinstance(opponent.state, KOState)) and jump:
        return -3.0 * env.dt

    if not isinstance(opponent.state, KOState):
        if abs(xo) < 7.0:
            if (xo > xp and player.body.velocity.x > 0) or abs(xp-xo) < 0.5:
                return 3.0 * env.dt
            if (xo < xp and player.body.velocity.x < 0) or abs(xp-xo) < 0.5:
                return 3.0 * env.dt
            if yo < yp and player.body.velocity.y < 0 and opponent.body.velocity.y < 0:
                return 3.0 * env.dt
    return 0.0


def self_preservation_reward(env: WarehouseBrawl) -> float:
    """Reward getting back to safety when in danger."""
    player: Player = env.objects["player"]
    
    x, y = player.body.position.x, player.body.position.y
    in_danger = y > 3.0 or abs(x) > 7.0 or (-2.0 < x < 2.0 and y > 0.5)
    
    if not in_danger:
        return 0.0
    
    reward = 0.0
    
    # Reward moving toward safety
    if y > 3.0:  # Too low
        # Reward upward velocity
        if player.body.velocity.y < 0:
            reward += 12.0
        
        # Reward using recovery
        if isinstance(player.state, AttackState) and hasattr(player.state, 'move_type'):
            if player.state.move_type == MoveType.RECOVERY:
                reward += 15.0
    
    # Reward moving toward platforms from gap
    if -2.0 < x < 2.0:
        # Moving left toward left platform
        if x < 0 and player.body.velocity.x < 0:
            reward += 5.0
        # Moving right toward right platform
        elif x > 0 and player.body.velocity.x > 0:
            reward += 5.0
    
    # Reward moving inward from edges
    if x < -7.0 and player.body.velocity.x > 0:
        reward += 5.0
    elif x > 7.0 and player.body.velocity.x < 0:
        reward += 5.0
    
    if player.is_on_floor():
        reward += 10.0
    
    return reward * env.dt

def off_screen_penalty(env: WarehouseBrawl) -> float:
    """Strong penalty for going off screen."""
    player: Player = env.objects["player"]
    
    reward = 0.0
    x, y = player.body.position.x, player.body.position.y
    
    # Escalating penalties based on danger
    if y > 7.0:  # Very far off screen
        reward -= 50.0 * env.dt
    elif y > 5.0:  # Off screen
        reward -= 20.0 * env.dt
    elif y > 4.2:  # Danger zone
        reward -= 8.0 * env.dt
    
    # Lateral boundaries
    if abs(x) > 8.0:  # Very far off
        reward -= 20.0 * env.dt
    elif abs(x) > 7.1:  # Off sides
        reward -= 7.0 * env.dt
    
    return reward

def map_safety_reward(env: WarehouseBrawl) -> float:
    """Stronger rewards for staying safe."""
    player: Player = env.objects["player"]
    x = player.body.position.x
    y = player.body.position.y

    reward = 0.0

    # STRONG punish for gap area
    if -2.0 < x < 2.0 and y > 0.5:
        reward -= 20.0 * env.dt  # Increased from 10.0

    # STRONG punish for falling
    if y > 4.0:
        reward -= 100.0 * env.dt  # Increased from 50.0

    # STRONG reward for safe platforms
    if (-7.0 < x < -2.0 and y < 1.5) or (2.0 < x < 7.0 and y < 1.5):
        reward += 10.0 * env.dt  # Increased from 5.0
    
    return reward

def input_spam_penalty(env: WarehouseBrawl) -> float:
    """Penalize pressing too many buttons at once."""
    player: Player = env.objects["player"]
    
    # Count number of active inputs
    active_inputs = (player.cur_action > 0.5).sum()
    
    if active_inputs > 3:
        return -5.0 * env.dt
    elif active_inputs > 4:
        return -10.0 * env.dt
    
    return 0.0

def damage_diff_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    return (opponent.damage_taken_this_frame - player.damage_taken_this_frame) * 0.5

def neutral_control_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    dx = abs(player.body.position.x - opponent.body.position.x)
    safe_range = 1.5 < dx < 3.0  # “whiff punish” zone
    if safe_range and player.is_on_floor():
        return 2.0 * env.dt
    return 0.0

def punish_window_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    if not hasattr(player, 'just_dodged_frame'):
        player.just_dodged_frame = -100
    
    reward = 0.0
    frame = env.steps
    
    # Detect successful dodge near attack
    if isinstance(opponent.state, AttackState) and isinstance(player.state, DodgeState):
        if abs(player.body.position.x - opponent.body.position.x) < 1.0:
            player.just_dodged_frame = frame
    
    # If we hit opponent within ~15 frames after dodge
    if opponent.damage_taken_this_frame > 0 and frame - player.just_dodged_frame < 15:
        reward += 20.0 * env.dt
    
    return reward

def survival_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    if player.stocks > 0:
        return 0.5 * env.dt  # small positive reward every frame alive
    return 0.0

def taunt_and_drop_penalty(env: WarehouseBrawl) -> float:
    """Penalize taunting. and dropping weapon"""
    player: Player = env.objects["player"]
    keys = player.input.key_status

    if keys["h"].held:
        return -10.0 * env.dt
    if keys["g"].held:
        return -10.0 * env.dt
    return 0.0

def action_consistency_reward(env: WarehouseBrawl) -> float:
    """Reward holding inputs instead of button mashing."""
    player: Player = env.objects["player"]
    
    # Track previous action
    if not hasattr(player, 'prev_action'):
        player.prev_action = player.cur_action.copy()
        return 0.0
    
    # Calculate how similar current action is to previous
    similarity = np.sum(player.cur_action == player.prev_action) / len(player.cur_action)
    
    player.prev_action = player.cur_action.copy()
    
    # Reward consistency (but not too much - we still want adaptability)
    if similarity > 0.7:
        return 1.0 * env.dt
    
    return 0.0

def win_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if opponent.stocks == 0:
        return 30 * env.dt
    return 0.0
    
def stay_on_stage_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    if abs(x) > 6.5:
        return -40.0 * env.dt
    if x > 0 and y > 0.85:
        return -15.0 * env.dt
    if x < 0 and y > 2.85:
        return -15.0 * env.dt
    return 0.0

def return_low_jumps_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    x = player.body.position.x

    if isinstance(player.state, InAirState):
        if hasattr(player.state, 'jumps_left') and hasattr(player.state, 'recoveries_left') and player.state.recoveries_left + player.state.jumps_left <= 1 and (abs(x) > 7.0 or abs(x) < 2.0):
            return -18.0 * env.dt 
    return 0.0
    
def damage_dealt_reward(env: WarehouseBrawl) -> float:
    opponent = env.objects["opponent"]
    return opponent.damage_taken_this_frame * env.dt 

def stock_loss_penalty(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    if player.stocks == 0:
        return -100.0 * env.dt
    if isinstance(player.state, KOState):
        return -50.0 * env.dt
    return 0.0
def reward_heavy_attack(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    if isinstance(player.state, AttackState) and player.state.move_type in [MoveType.NSIG, MoveType.DSIG, MoveType.SSIG]:
        if opponent.damage_taken_this_frame > 0 and opponent.damage_taken_this_stock >= 55:
            return 3.0 * env.dt

    return 0.0

def conflicting_keys_penalty(env: WarehouseBrawl) -> float:
    """Penalize holding opposite directional keys at the same time."""
    player: Player = env.objects["player"]
    keys = player.input.key_status

    penalty = 0.0

    # Horizontal conflict: A (left) and D (right)
    if keys["A"].held and keys["D"].held:
        penalty -= 2.0 * env.dt

    # Vertical conflict: W (up) and S (down)
    if keys["W"].held and keys["S"].held:
        penalty -= 2.0 * env.dt

    return penalty

def low_jump_penalty(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    x = player.body.position.x

    if isinstance(player.state, InAirState):
        if hasattr(player.state, 'jumps_left') and player.state.jumps_left == 0:
            return -1.2 * env.dt 
    return 0.0

def wiff_penalty(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    if isinstance(player.state, KOState):
        return -10.0 * env.dt
    return 0.0

def attack_frequency_reward(env: WarehouseBrawl) -> float:
    """Reward attacking when close to opponent."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Proper distance check
    distance = math.sqrt(
        (player.body.position.x - opponent.body.position.x)**2 +
        (player.body.position.y - opponent.body.position.y)**2
    )
    
    if distance > 1.5:
        return 0.0
    
    if not hasattr(player, 'frames_since_attack'):
        player.frames_since_attack = 0
    
    if isinstance(player.state, AttackState):
        player.frames_since_attack = 0
        return 1.0 * env.dt  # Reduced from 1.5
    else:
        player.frames_since_attack += 1
    
    if player.frames_since_attack > 90:
        return -3.0 * env.dt  # Reduced from -2.0 but still significant
    
    return 0.0

def self_destruct_penalty(env: WarehouseBrawl) -> float:
    """Heavy penalty for dying without opponent dealing much damage."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    if isinstance(player.state, KOState):
        return -12.0 * env.dt
    
    return 0.0
    
def parry_god_reward(env: WarehouseBrawl) -> float:
    """Perfect dodge timing, then punish hard."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Initialize tracking
    if not hasattr(player, 'parry_state'):
        player.parry_state = {
            'just_dodged': False,
            'dodge_frame': -100,
            'punish_window_active': False
        }
    
    state = player.parry_state
    current_frame = env.steps
    
    distance = math.sqrt(
        (player.body.position.x - opponent.body.position.x)**2 +
        (player.body.position.y - opponent.body.position.y)**2
    )
    
    reward = 0.0
    
    # PHASE 1: Opponent is attacking - DODGE NOW
    if isinstance(opponent.state, AttackState) and distance < 1.2:
        if isinstance(player.state, DodgeState):
            reward += 20.0 * env.dt  # HUGE reward for dodging during attack
            state['just_dodged'] = True
            state['dodge_frame'] = current_frame
            state['punish_window_active'] = True
        elif not isinstance(player.state, DodgeState):
            # You should be dodging!
            reward -= 10.0 * env.dt
    
    # PHASE 2: Just dodged - PUNISH NOW (15 frame window)
    frames_since_dodge = current_frame - state['dodge_frame']
    if state['punish_window_active'] and frames_since_dodge < 15:
        if isinstance(player.state, AttackState):
            reward += 30.0 * env.dt  # MASSIVE reward for punish
            
            # Extra bonus if we hit
            if opponent.damage_taken_this_frame > 0:
                reward += 50.0 * env.dt  # JACKPOT!
                state['punish_window_active'] = False  # Reset after successful punish
        else:
            # You have a punish window, USE IT!
            reward += 5.0 * env.dt  # Small reward for being ready
    elif frames_since_dodge >= 15:
        state['punish_window_active'] = False
    
    # PHASE 3: Neutral game - Stay defensive, bait attacks
    if not isinstance(opponent.state, AttackState) and not state['punish_window_active']:
        # Stay at mid range (baiting distance)
        if 1.5 < distance < 3.0:
            reward += 8.0 * env.dt  # Good spacing
        
        # DON'T attack randomly - wait for them
        if isinstance(player.state, AttackState):
            reward -= 15.0 * env.dt  # Punish random attacks
        
        # Stay grounded and ready
        if player.is_on_floor() and isinstance(player.state, StandingState):
            reward += 3.0 * env.dt
    
    return reward


def timeout_runner_reward(env: WarehouseBrawl) -> float:
    """When ahead, run away and stall."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    px, py = player.body.position.x, player.body.position.y
    ox, oy = opponent.body.position.x, opponent.body.position.y
    
    # Calculate if we're winning
    stock_advantage = player.stocks - opponent.stocks
    damage_advantage = opponent.damage_taken_this_stock - player.damage_taken_this_stock
    
    # Are we ahead?
    winning = stock_advantage > 0 or \
              (stock_advantage == 0 and damage_advantage > 30)
    
    if not winning:
        return 0.0  # Don't run if not winning
    
    reward = 0.0
    horizontal_distance = abs(px - ox)
    
    # MAXIMIZE distance from opponent
    if horizontal_distance > 4.0:
        reward += 15.0 * env.dt  # HUGE reward for staying far
    elif horizontal_distance > 3.0:
        reward += 8.0 * env.dt
    elif horizontal_distance < 2.0:
        reward -= 10.0 * env.dt  # Penalty for being close
    
    # Actively move AWAY from opponent
    moving_away = (px < ox and player.body.velocity.x < 0) or \
                  (px > ox and player.body.velocity.x > 0)
    
    if moving_away:
        reward += 10.0 * env.dt
    else:
        reward -= 5.0 * env.dt  # You should be running!
    
    # NEVER attack when running away
    if isinstance(player.state, AttackState):
        reward -= 30.0 * env.dt  # Heavy penalty
    
    # Use platforms to your advantage (higher is safer)
    if py < 1.0:  # On a high platform
        reward += 5.0 * env.dt
    
    # If opponent is chasing, lead them into bad positions
    if ox > 6.5 or ox < -6.5:  # Opponent near edge
        if horizontal_distance > 3.0:  # And we're safe
            reward += 10.0 * env.dt  # They're in danger, we're not
    
    return reward


def adaptive_gameplan_reward(env: WarehouseBrawl) -> float:
    """Switches between Parry God and Timeout King based on game state."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    stock_advantage = player.stocks - opponent.stocks
    damage_advantage = opponent.damage_taken_this_stock - player.damage_taken_this_stock
    
    # Winning condition: up a stock OR up 30%+ damage
    winning = stock_advantage > 0 or \
              (stock_advantage == 0 and damage_advantage > 30)
    
    if winning:
        # RUN AWAY MODE
        return timeout_runner_reward(env)
    else:
        # PARRY GOD MODE
        return parry_god_reward(env)


def defensive_positioning_reward(env: WarehouseBrawl) -> float:
    """Stay in safe positions with escape routes."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    px, py = player.body.position.x, player.body.position.y
    
    reward = 0.0
    
    # Prefer main platforms (safe positions)
    if (-6.5 < px < -2.5) or (2.5 < px < 6.5):
        if py < 1.0:  # On platform, not in air
            reward += 5.0 * env.dt
    
    # Avoid corners (can't escape)
    if abs(px) > 6.8:
        reward -= 8.0 * env.dt
    
    # Avoid center gap (can fall through)
    if -2.0 < px < 2.0 and py > 0.5:
        reward -= 10.0 * env.dt
    
    # Face opponent (ready to react)
    dx = opponent.body.position.x - px
    facing_opponent = (dx > 0 and player.facing == Facing.RIGHT) or \
                     (dx < 0 and player.facing == Facing.LEFT)
    
    if facing_opponent:
        reward += 2.0 * env.dt
    
    return reward


def punish_aggression_reward(env: WarehouseBrawl) -> float:
    """Extra reward for making opponent pay for attacking."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # If opponent whiffed an attack (they attacked, we didn't get hit)
    if isinstance(opponent.state, AttackState):
        if player.damage_taken_this_frame == 0:
            # They missed! Capitalize!
            if isinstance(player.state, AttackState):
                if opponent.damage_taken_this_frame > 0:
                    # We punished their whiff!
                    return 40.0 * env.dt
    
    return 0.0


def never_attack_first_reward(env: WarehouseBrawl) -> float:
    """Punish throwing out attacks without opponent committing first."""
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Are we winning (should be running away)?
    stock_advantage = player.stocks - opponent.stocks
    damage_advantage = opponent.damage_taken_this_stock - player.damage_taken_this_stock
    winning = stock_advantage > 0 or (stock_advantage == 0 and damage_advantage > 30)
    
    if winning:
        return 0.0  # timeout_runner handles this
    
    # In neutral, don't attack unless opponent is committed
    if isinstance(player.state, AttackState):
        if not isinstance(opponent.state, AttackState):
            # We attacked while they're neutral - BAD
            return -12.0 * env.dt
    
    return 0.0


# ===== UPDATED GEN_REWARD_MANAGER =====

def gen_reward_manager():
    reward_functions = {
        # ===== CORE STRATEGY (Highest Priority) =====
        'adaptive_gameplan': RewTerm(func=adaptive_gameplan_reward, weight=1.0),  # Switches modes
        
        # ===== SUPPORTING REWARDS =====
        'defensive_positioning': RewTerm(func=defensive_positioning_reward, weight=0.8),
        'punish_aggression': RewTerm(func=punish_aggression_reward, weight=1.0),
        'never_attack_first': RewTerm(func=never_attack_first_reward, weight=0.6),
        
        # ===== DAMAGE FEEDBACK (Moderate) =====
        'damage_dealt_reward': RewTerm(func=damage_dealt_reward, weight=3.0),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),
        'combo_reward': RewTerm(func=combo_reward, weight=1.5),
        
        # ===== SAFETY (Important) =====
        'return_to_platform_gentle': RewTerm(func=return_to_platform_gentle, weight=2.0),
        'off_screen_penalty': RewTerm(func=off_screen_penalty, weight=1.5),
        
        # ===== MINOR SHAPING =====
        'face_right_direction_reward': RewTerm(func=face_right_direction_reward, weight=0.2),
        'taunt_and_drop_penalty': RewTerm(func=taunt_and_drop_penalty, weight=1.0),
    }
    
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=500)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=80)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
    }
    
    return RewardManager(reward_functions, signal_subscriptions)


# ===== TRAINING CONFIGURATION =====

if __name__ == '__main__':
    # Start fresh
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)
    
    # Reward manager
    reward_manager = gen_reward_manager()
    
    # Self-play handler
    selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
    
    # Save settings
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=200_000,
        max_saved=40,
        save_path='checkpoints',
        run_name='parry_god_timeout_king_2',
        mode=SaveHandlerMode.FORCE
    )
    
    # Opponent mix: Focus on learning to counter aggressive play
    opponent_specification = {
        'self_play': (3, selfplay_handler),      # 30% - learn to fight yourself
        'based_agent': (7, partial(BasedAgent)), # 70% - master the basics
    }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)
    
    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=3_000_000,
        train_logging=TrainLogging.PLOT
    )
    

# '''
# Add your dictionary of RewardFunctions here using RewTerms
# '''
# def gen_reward_manager():
#     reward_functions = {

#         'approach_opponent_reward': RewTerm(func=approach_opponent_reward, weight=1),
#         'smart_attack_reward': RewTerm(func=smart_attack_reward, weight=1),

#         #'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=3),


#         'string_reward': RewTerm(func=string_reward, weight=1.0),

#         'taunt_and_drop_penalty': RewTerm(func=taunt_and_drop_penalty, weight=1.0),



#         'face_right_direction_reward': RewTerm(func=face_right_direction_reward, weight=1),



#         'attack_frequency_reward': RewTerm(func=attack_frequency_reward, weight=1),
#         'return_to_platform_gentle': RewTerm(func=return_to_platform_gentle, weight=2),
#         'self_destruct_penalty': RewTerm(func=self_destruct_penalty, weight=1),



#     }
#     signal_subscriptions = {
#         'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=150)),
#         'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=30)),
#         # 'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
#         'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
#         # 'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=15))
#     }
#     return RewardManager(reward_functions, signal_subscriptions)

# # -------------------------------------------------------------------------
# # ----------------------------- MAIN FUNCTION -----------------------------
# # -------------------------------------------------------------------------
# '''
# The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
# '''
# if __name__ == '__main__':
#     # Create agent
#     my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

#     # Start here if you want to train from scratch. e.g:
#     #my_agent = RecurrentPPOAgent()

#     # Start here if you want to train from a specific timestep. e.g:
#     #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

#     # Reward manager
#     reward_manager = gen_reward_manager()
#     # Self-play settings
#     selfplay_handler = SelfPlayRandom(
#         partial(type(my_agent)), # Agent class and its keyword arguments
#                                  # type(my_agent) = Agent class
#     )

#     # Set save settings here:
#     save_handler = SaveHandler(
#         agent=my_agent, # Agent to save
#         save_freq=1_000_000, # Save frequency
#         max_saved=40, # Maximum number of saved models
#         save_path='checkpoints', # Save path
#         run_name='experiment_based_Minimal_v17',
#         mode=SaveHandlerMode.FORCE  # Save mode, FORCE or RESUME
#     )

#     # Set opponent settings here:
#     opponent_specification = {
#                     'self_play': (2, selfplay_handler),
#                     'SB3_agent': (2, partial(SB3Agent)),
#                     'clockwork_agent': (2, partial(ClockworkAgent)),
#                     'based_agent': (2, partial(BasedAgent)),
#                     'RecurrentPPO_agent': (2, partial(RecurrentPPOAgent)),
#                 }
#     opponent_cfg = OpponentsCfg(opponents=opponent_specification)

#     train(my_agent,
#         reward_manager,
#         save_handler,
#         opponent_cfg,
#         CameraResolution.LOW,
#         train_timesteps=6_000_000,
#         train_logging=TrainLogging.PLOT
#     )
