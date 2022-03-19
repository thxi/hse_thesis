import pandas as pd
import numpy as np
from bandits.agents import EpsilonGreedyAgent

df = pd.DataFrame(np.ones(shape=(10, 10)))
print(df)

a = EpsilonGreedyAgent()
a.take_action()
