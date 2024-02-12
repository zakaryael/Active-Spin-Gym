from icecream import ic
from tqdm import tqdm
from envs import make
import time


def main():

    env = make("NavigateT")
    print(env.simulation.lattice)

    for i in (range(2000)):
        try:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print(f"\ndone at step {i}, reward = {reward}, info = {info}")
                break
            env.render("console")
        except Exception as e:
            print("An error occurred:", e)
            raise e


if __name__ == "__main__":
    main()
