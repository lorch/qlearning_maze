from Maze import Maze
from Robot import Robot

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

## todo: 创建迷宫并展示
# maze = Maze(maze_size=(10, 10), trap_number=5)

# robot = Robot(maze)  # 记得将 maze 变量修改为你创建迷宫的变量名
# robot.set_status(learning=True, testing=False)
# print(robot.update())

# maze
# table = {'l': {'u': -10., 'l': 50., "r": -30., "d": -0.1}}
# ['hit_wall'] = -10.
# table['l']['destination'] = 50.
# table['l']['trap'] = -30.
# table['l']['default'] = -0.1
# table['r']['hit_wall'] = -11.
# table['r']['destination'] = 51.
# table['r']['trap'] = -31.
# table['r']['default'] = -0.11
# table['u']['hit_wall'] = -8.
# table['u']['destination'] = 20.
# table['u']['trap'] = -10.
# table['u']['default'] = -1.1
# table['d']['hit_wall'] = -10.5
# table['d']['destination'] = 50.5
# table['d']['trap'] = -30.5
# table['d']['default'] = -0.6
# max_q = max(list(table['l'].values()))
# print(max_q)
# max_index = list(table['l'].values()).index(max_q)
# print(max_index)
# print(list(table['l'].keys())[max_index])

## 可选的参数：
epoch = 20

epsilon0 = 0.3
alpha = 0.5
gamma = 0.9

maze_size = (6, 6)
trap_number = 1

from Runner import Runner

g = Maze(maze_size=maze_size, trap_number=trap_number)
r = Robot(g, alpha=alpha, epsilon0=epsilon0, gamma=gamma)
r.set_status(learning=True)

runner = Runner(r, g)
runner.run_training(epoch, display_direction=True)
runner.plot_results()
# runner.generate_movie(filename="final1.mp4")

# print(0.9 * 0.9)
