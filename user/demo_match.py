from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import UserInputAgent, CustomAgent, BasedAgent, MLPExtractor, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
from stable_baselines3 import PPO
import pygame
pygame.init()

trained_model_path = "checkpoints/experiment_based_Minimal_v9/rl_model_10003500_steps"
my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor, file_path=trained_model_path)

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = BasedAgent()

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='based-t3.mp4'
)
