"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
from progbar import ProgBar

def update():
    max_episode = 30
    pb = ProgBar(max_episode)
    for episode in range(max_episode):
        try:
            # initial observation
            observation = env.reset()
            pb.show_progbar()
            while True:
                # fresh env
                #env.render()

                # RL choose action based on observation
                action = RL.choose_action(str(observation))

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)
    
                # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_))

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    break
        except KeyboardInterrupt:
            env.destroy()
            break

    # end of game
    print('train over, let play!')
   
    # make window visible.
    env.deiconify()

    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        env.render()
        action = RL.exploit_action(str(obs))
        obs_, reward, done = env.step(action)
        obs = obs_
        total_reward += reward
        steps +=1
        if done:
            break
    print('Game over! spend {0} steps, total reward={1}'.format(steps,total_reward))

    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
