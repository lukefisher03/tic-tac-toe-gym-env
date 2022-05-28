# import gym
# from gym import spaces

# observation_space = spaces.MultiDiscrete([[3,3,3] for col in range(3)])

# print(observation_space.sample())

board = [
    [2,0,1],
    [0,2,0],
    [0,0,2]
]

def checkBoard(board) -> int:
    for p in [1,2]:

        
        for row in board:
            if row == [p,p,p]:
                print(f'p{p} wins by horizontal row')
                return p

        for i in range(3):
            col = []
            for row in board:
                col.append(row[i])
            if col == [p,p,p]:
                print(f'p{p} wins by vertical column')
                return p

        i = 0
        diag = []
        for row in board:
            diag.append(row[i])
            i+=1
        if diag == [p,p,p]:
            print(f'p{p} wins by diagonal')
            return p

        i = 2
        diag = []
        for row in board:
            diag.append(row[i])
            i -= 1
        if diag == [p,p,p]:
            print(f'p{p} wins by diagonal')
            return p