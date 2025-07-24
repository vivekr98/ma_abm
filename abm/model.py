from mesa import Model
from .agents import WorkerAgent
import random

class GigEconomyModel(Model):
    def __init__(self, n):
        super().__init__()
        self.num_agents = n

        # Create agents
        WorkerAgent.create_agents(model = self, n=n)

    def step(self):
        self.agents.shuffle_do("exchange")
