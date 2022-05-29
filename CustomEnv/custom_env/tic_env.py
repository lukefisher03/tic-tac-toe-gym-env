import gym
from gym import spaces

'''
This is a custom OpenAI gym environment for simulating and training agents to play
Tic Tac Toe.

This game is played on an empty 3x3 board(represented as a 2d list). 
    Observation Returned to Agent: 3x3 grid describing whether or not a tile is empty(0), has an x(1),
                                   or has an o(2)
    Action Space: [x,y] -> coordinates to the square the AI would like to place the piece. 

    Rewards: (WIP) The reward space returns 0 for any move. Losing the game(terminal state) yields -5 and winning
              the game yields +5
'''


#Returns whether or not someone has won the game. Returns an int with 0 being non-winning state, 1: x wins, 2: o wins.
def checkBoardState(board) -> int:
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
            print(f'p{p} wins by diagonal (top right to bottom left)')
            return p

        i = 2
        diag = []
        for row in board:
            diag.append(row[i])
            i -= 1
        if diag == [p,p,p]:
            print(f'p{p} wins by diagonal (bottom left to top right)')
            return p

class TicTacToe(gym.Env):
    metadata = {
        'render.modes' : ['human']
    }

    def __init__(self) -> None:
        super(TicTacToe, self).__init__()

        self.action_space = spaces.MultiDiscrete([[0,1,2],[0,1,2]])
        self.observation_space = spaces.MultiDiscrete([[3,3,3] for col in range(3)])
        self.reward_range = (-5, 5)
        #initalize an empty game board on each new episode.
        self.gameBoard = [[0,0,0] for col in range(3)]

        '''
        side note: Initally the gameBoard would be declared in the global space along with the algorithm. But in python
        class attributes and variables are accessible everywhere. This makes the implementation of a two player game easier,
        the agent will take its turn, and then a second agent will take its turn switching back and forth modifying the gameBoard
        instantiated with each episode.   
        '''
    def step(self, action): 
        #For sake of simplicity, our agent always plays as x's. Whoever goes first will be determined randomly before the start of the game.
        done = False
        reward = 0
        x,y = action
        if self.gameBoard[x,y] != 0:#end the game early and pass extremely negative reward for placing a pieces that's already taken. (cheating)
            done = True
            reward = -1000
        self.gameBoard[x,y] = 1 #Place the agent's piece on the board.
        
        winState = checkBoardState(self.gameBoard)
        if winState == 1:
            reward = 5
            done = True
        elif winState == 2:
            reward = -5
            done = True
        
        return self.gameBoard, reward, done, {}

    def reset(self):
        self.gameBoard = [[0,0,0] for col in range(3)]
        return self.gameBoard

    def render(self):
        print(self.gameBoard())

    def close(self):
        print('Shutting down the environment...')