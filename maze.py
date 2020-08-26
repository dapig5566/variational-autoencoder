import numpy as np
import curses
import time

class Agent():
    def __init__(self, x, y, h_limit, w_limit):
        self.x = x
        self.y = y
        self.h_limit = h_limit
        self.w_limit = w_limit

    def act(self, action):
        x_a, y_a = 0, 0
        if action == 0:
            x_a = np.minimum(self.x+1, self.h_limit-1)
            y_a = self.y
        elif action == 1:
            x_a = np.maximum(self.x-1, 0)
            y_a = self.y
        elif action == 2:
            y_a = np.minimum(self.y+1, self.w_limit-1)
            x_a = self.x
        elif action == 3:
            y_a = np.maximum(self.y-1, 0)
            x_a = self.x
        elif action == 5:
            x_a = self.x
            y_a = self.y
        return (x_a, y_a)

    def apply_position(self, x, y):
        self.x = x
        self.y = y
    @property
    def position(self):
        return (self.x, self.y)

def argmax(mat):
    max_val = -np.inf
    x, y, z = 0, 0, 0
    for i, a in enumerate(mat):
        for j, b in enumerate(a):
            if b > max_val:
                max_val = b
                x, y = i, j
    return [x, y]

def get_position(agents):
    pos = [(i.x, i.y) for i in agents]
    return pos

mapfile = open('map', 'r')
m = []
while True:
    r = mapfile.readline()
    if r=='':
        break
    if r[-1] == '\n':
        m.append(list(r[:-1]))
    else:
        m.append(list(r))





target_position = {}
for i, row in enumerate(m):
    for j, c in enumerate(row):
        if c == '1' or c == '2':
            target_position[c] = (i, j)



# start_position = {'A': (3, 0), 'B':(3, 9)}

epochs = 10000
h, w = len(m), len(m[0])

q_tab = np.zeros([h, w, h, w, 4, 4])
epi = 1.0
lr = 0.01

for epoch in range(epochs):
    if epoch == epochs-1:
        start_position = {'A': (3, 0), 'B': (3, 9)}
    else:
        start_position = []
        while len(start_position) < 2:
            x = np.random.choice(4)
            y = np.random.choice(10)
            if (x, y) not in target_position.values():
                start_position.append((x, y))
        start_position = {i: j for i, j in zip(['A', 'B'], start_position)}
    ep_score = 0
    maze = np.array(m.copy())
    agents = [Agent(*i, h, w) for i in start_position.values()]
    end = False
    while not end:
        if epoch == epochs-1:
            show = maze.copy()
            for agent, i in zip(agents, ['A', 'B']):
                show[agent.x, agent.y] = i
            print(show)

        pos = get_position(agents)

        action = argmax(q_tab[pos[0][0], pos[0][1], pos[1][0], pos[1][1]])
        if np.random.uniform() < epi and epoch < epochs - 1:
            action[0] = np.random.choice(4)
        if np.random.uniform() < epi and epoch < epochs - 1:
            action[1] = np.random.choice(4)
        after_position = [agent.act(act) for agent, act in zip(agents, action)]
        if after_position[0] == after_position[1]:
            after_position[0] = agents[0].position
            after_position[1] = agents[1].position
        reward = 0
        for agent, position, key in zip(agents, after_position, sorted(target_position)):
            if position == target_position[key]:
                reward += 50
                end = True
            else:
                reward -= 0.5

        ep_score += reward
        if epoch == epochs - 1:
            print(reward)

        if end:
            q_tab[pos[0][0], pos[0][1], pos[1][0], pos[1][1], action[0], action[1]] = \
                q_tab[pos[0][0], pos[0][1], pos[1][0], pos[1][1], action[0], action[1]] * (1 - lr) + lr * reward
        else:
            q_tab[pos[0][0], pos[0][1], pos[1][0], pos[1][1], action[0], action[1]] = \
                q_tab[pos[0][0], pos[0][1], pos[1][0], pos[1][1], action[0], action[1]]*(1-lr) + \
                lr*(reward + 0.99 * np.max(q_tab[after_position[0][0], after_position[0][1], after_position[1][0], after_position[1][1]]))

        for agent, position in zip(agents, after_position):
            agent.apply_position(*position)
        if epi > 0.1:
            epi = 1 - (0.9 / epochs) * epoch
    if epoch % 100 == 0:
         print('ep {}: score: {}'.format(epoch, ep_score))
# std = curses.initscr()
# for epoch in range(1):
#     ep_score = 0
#     maze = np.array(m.copy())
#     agent_x, agent_y = s_x, s_y
#     end = False
#     while not end:
#         show = maze.copy()
#         show[agent_x, agent_y] = '@'
#         std.addstr(0, 0, str(show))
#         std.refresh()
#         action = np.argmax(q_tab[agent_x, agent_y])
#         x_a, y_a = 0, 0
#         if action == 0:
#             x_a = np.minimum(agent_x+1, h-1)
#             y_a = agent_y
#         elif action == 1:
#             x_a = np.maximum(agent_x-1, 0)
#             y_a = agent_y
#         elif action == 2:
#             y_a = np.minimum(agent_y+1, w-1)
#             x_a = agent_x
#         elif action == 3:
#             y_a = np.maximum(agent_y-1, 0)
#             x_a = agent_x
#
#         if maze[x_a, y_a] == '#':
#             reward = 5
#             end = True
#         elif maze[x_a, y_a] == 'A':
#             maze[x_a, y_a] = '.'
#             reward = 1
#         else:
#             reward = 0
#         ep_score += reward
#         std.addstr(10, 0, str(ep_score))
#         std.refresh()
#         time.sleep(0.5)
#         agent_x, agent_y = x_a, y_a
#     show = maze.copy()
#     show[agent_x, agent_y] = '@'
#     std.addstr(0, 0, str(show))
#     std.refresh()
# time.sleep(5)
# curses.endwin()


