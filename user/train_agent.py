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
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.007)
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

def passive_survival_reward(env: WarehouseBrawl) -> float:
    """Main survival incentive: stay on platforms, avoid falling."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    # Safe zones (main platforms)
    safe_zone = (-7.0 < x < -2.0 and y < 3.2) or (2.0 < x < 7.0 and y < 1.2)

    reward = 0.0
    if safe_zone:
        reward += 8.0 * env.dt  # reward calm presence on safe ground
    elif -2.0 < x < 2.0 and y < 0.8:
        reward += 3.0 * env.dt  # reward for standing on the middle platform
    else:
        reward -= 10.0 * env.dt  # discourage being off safe ground

    return reward

def void_avoidance_penalty(env: WarehouseBrawl) -> float:
    """Strong penalty for hovering above or below the central void."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    # If directly above or falling into the void zone
    if -2.0 < x < 2.0 and y > 0.5:
        return -20.0 * env.dt
    if y > 3.5:
        return -60.0 * env.dt  # falling too low = deadly

    return 0.0

def stable_movement_reward(env: WarehouseBrawl) -> float:
    """Encourages gentle, stable movement instead of chaotic running/jumping."""
    player: Player = env.objects["player"]
    vx, vy = abs(player.body.velocity.x), abs(player.body.velocity.y)

    reward = 0.0
    if vx < 1.5 and vy < 1.0:
        reward += 2.0 * env.dt  # smooth controlled movement
    elif vy > 3.0:
        reward -= 6.0 * env.dt  # reckless jumps
    return reward

def recovery_focus_reward(env: WarehouseBrawl) -> float:
    """Encourages upward motion and recovery when off stage."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    reward = 0.0
    # When near edges or below platforms, reward upward motion
    if abs(x) > 6.5 or y > 2.5:
        if player.body.velocity.y < 0:
            reward += 8.0 * env.dt
        else:
            reward -= 5.0 * env.dt
    return reward

def anti_aggression_penalty(env: WarehouseBrawl) -> float:
    """Discourage unnecessary attacking."""
    player: Player = env.objects["player"]

    if isinstance(player.state, AttackState):
        return -4.0 * env.dt
    return 0.0

def grounded_bonus_reward(env: WarehouseBrawl) -> float:
    """Reward for staying grounded instead of jumping unnecessarily."""
    player: Player = env.objects["player"]

    if player.is_on_floor():
        return 3.0 * env.dt
    return -1.0 * env.dt

def edge_distance_reward(env: WarehouseBrawl) -> float:
    """Reward keeping distance from the outer edges."""
    player: Player = env.objects["player"]
    x = abs(player.body.position.x)

    if 5.0 < x < 6.5:
        return 1.5 * env.dt  # cautious near edge
    elif x >= 6.5:
        return -8.0 * env.dt  # too close to edge
    return 0.0

def self_preservation_reward(env: WarehouseBrawl) -> float:
    """Reward getting back to safety when near death zones."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    reward = 0.0

    # Detect danger (off stage or below platform height)
    in_danger = (y > 3.0) or (abs(x) > 7.0) or (-2.0 < x < 2.0 and y > 0.5)

    if in_danger:
        # Reward moving toward center or upward
        if y > 3.0 and player.body.velocity.y < 0:
            reward += 12.0  # ascending = good
        if x < -7.0 and player.body.velocity.x > 0:
            reward += 6.0   # moving back inward
        if x > 7.0 and player.body.velocity.x < 0:
            reward += 6.0
        if -2.0 < x < 2.0:
            if x < 0 and player.body.velocity.x < 0:
                reward += 4.0
            elif x > 0 and player.body.velocity.x > 0:
                reward += 4.0
    else:
        reward += 2.0  # small bonus for being safe

    return reward * env.dt

def off_screen_penalty(env: WarehouseBrawl) -> float:
    """Strong penalty for going off or near off-screen zones."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y
    penalty = 0.0

    # Vertical danger
    if y > 7.0:
        penalty -= 60.0 * env.dt
    elif y > 5.0:
        penalty -= 30.0 * env.dt
    elif y > 4.0:
        penalty -= 10.0 * env.dt

    # Horizontal danger
    if abs(x) > 8.0:
        penalty -= 40.0 * env.dt
    elif abs(x) > 7.0:
        penalty -= 10.0 * env.dt

    return penalty

def map_safety_reward(env: WarehouseBrawl) -> float:
    """Reward staying on platforms; penalize dangerous areas."""
    player: Player = env.objects["player"]
    x, y = player.body.position.x, player.body.position.y

    reward = 0.0

    # Main platforms
    if (-7.0 < x < -2.0 and y < 3.2) or (2.0 < x < 7.0 and y < 1.2):
        reward += 8.0 * env.dt

    # Middle moving platform
    elif -2.0 < x < 2.0 and y < 0.8:
        reward += 3.0 * env.dt

    # Void or falling zones
    elif -2.0 < x < 2.0 and y > 0.8:
        reward -= 20.0 * env.dt
    elif y > 4.0:
        reward -= 60.0 * env.dt

    return reward

def survival_reward(env: WarehouseBrawl) -> float:
    """Small ongoing reward for staying alive."""
    player: Player = env.objects["player"]
    if player.stocks > 0:
        return 0.6 * env.dt
    return 0.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """Penalty when the player dies (knockout)."""
    if agent == 'player':
        return -100.0
    else:
        return 0.0


'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        # --- Safety and Survival ---
        'passive_survival_reward': RewTerm(func=passive_survival_reward, weight=3.0),
        'void_avoidance_penalty': RewTerm(func=void_avoidance_penalty, weight=3.5),
        'recovery_focus_reward': RewTerm(func=recovery_focus_reward, weight=2.8),
        'edge_distance_reward': RewTerm(func=edge_distance_reward, weight=2.5),
        'grounded_bonus_reward': RewTerm(func=grounded_bonus_reward, weight=2.0),
        'stable_movement_reward': RewTerm(func=stable_movement_reward, weight=1.8),

        # --- Anti-Aggression ---
        'anti_aggression_penalty': RewTerm(func=anti_aggression_penalty, weight=2.5),

        # --- Keep from dying ---
        'self_preservation_reward': RewTerm(func=self_preservation_reward, weight=3.5),
        'off_screen_penalty': RewTerm(func=off_screen_penalty, weight=4.0),
        'map_safety_reward': RewTerm(func=map_safety_reward, weight=3.0),

        # --- Optional mild incentive for survival ---
        'survival_reward': RewTerm(func=survival_reward, weight=0.5),
    }

    signal_subscriptions = {
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=50)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=500_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='passive-nov1-715pm',
        mode=SaveHandlerMode.FORCE  # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0, partial(ConstantAgent)),
                    'based_agent': (2, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=1_000_000,
        train_logging=TrainLogging.PLOT
    )
