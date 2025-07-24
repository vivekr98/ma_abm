from mesa import Agent
import random

class WorkerAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.rating = random.uniform(3.0, 5.0)

    def step(self):
        # Workers select jobs in order of their rating (handled by model)
        pass

    def exchange(self):
        if self.rating > 4.0:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is not None:
                other_agent.rating += 0.1
                self.rating -= 0.1