from envs.continuous.channel import Channel
from tqdm import tqdm

env = Channel(width=50, height=25, density=0.3, control_interval=1e-2)
_ = env.reset()

for i in tqdm(range(100000)):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    env.render()
