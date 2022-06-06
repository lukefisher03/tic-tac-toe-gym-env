import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO


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
INVALID_MOVE_REWARD = -100
VALID_MOVE_REWARD = 1
WIN_REWARD = 10
LOSS_REWARD = -10
TIE_REWARD = -5

#Returns whether or not someone has won the game. Returns an int with 0 being non-winning state, 1: x wins, 2: o wins, 3: tie.
def processBoardState(board) -> int:
    #convert board to 2d array
    board = np.reshape(board, (3,3))
    for p in [1,2]:
        for row in board:
            if np.array_equal(row, [p,p,p]):
                #print(f'p{p} wins by horizontal row')
                return p

        for i in range(3):
            col = []
            for row in board:
                col.append(row[i])
            if np.array_equal(col, [p,p,p]):
                #print(f'p{p} wins by vertical column')
                return p

        i = 0
        diag = []
        for row in board:
            diag.append(row[i])
            i+=1
        if np.array_equal(diag, [p,p,p]):
            #print(f'p{p} wins by diagonal (top right to bottom left)')
            return p

        i = 2
        diag = []
        for row in board:
            diag.append(row[i])
            i -= 1
        if np.array_equal(diag, [p,p,p]):
            #print(f'p{p} wins by diagonal (bottom left to top right)')
            return p

    if (board == 0).sum() == 0:
        return 3
    return 0

class TicTacToe(gym.Env):
    metadata = {
        'render.modes' : ['human']
    }

    def __init__(self, player2='') -> None:
        super(TicTacToe, self).__init__()

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3 for space in range(9)]) #0-8 spaces on tic-tac-toe board
        #initalize an empty game board on each new episode.
        self.gameBoard = np.array([0 for space in range(9)])
        self.p2 = player2
        self.wins = 0
        self.invalidMoves = 0
        self.ties = 0
        self.losses = 0

        if self.p2.agent == 2:
            self.p2Model = PPO.load('./models/PPO-OPTIMIZER')

        '''
        side note: Initally the gameBoard would be declared in the global space along with the algorithm. But in python
        class attributes and variables are accessible everywhere. This makes the implementation of a two player game easier,
        the agent will take its turn, and then a second agent will take its turn switching back and forth modifying the gameBoard
        instantiated with each episode.   
        '''
    def step(self, a): 
        '''
        The action space is presented as a singular scalar value representing a space on a flattened tic-tac-toe board. In this case it's our job to provide the correct
        rewards based on the given action. Primarily, we need to give guidelines for our agent. Reward it for winning, penalize it for losing, and heavily penalize it for cheating.
        Additionally, another scenario is a tie. By defualt we could pass a neutral 0 reward. However, passing a mildly negative reward could communicate that a tie is a 
        suboptimal state. By the credit assignment problem we can pass an extremely mild positive reward for simply making valid moves as they are necessary for achieving a 
        winning state.
        '''

        if self.gameBoard[a] != 0:
            print('INVALID MOVE')
            self.invalidMoves += 1
            return self.gameBoard, INVALID_MOVE_REWARD, True, {}

        self.gameBoard[a] = 1
        if self.p2.agent == 1:
            self.render()
        winner = processBoardState(self.gameBoard)
        if winner == 1:
            print('WIN')
            self.wins += 1
            return self.gameBoard, WIN_REWARD, True, {}
        elif winner == 3:
            print('TIE')
            self.ties += 1
            return self.gameBoard, TIE_REWARD, True, {}
        
        if self.p2.agent == 0:
            self.p2.move(self.gameBoard)
        elif self.p2.agent == 1:
            self.p2.moveHuman(self.gameBoard)
        elif self.p2.agent == 2:
            self.p2.moveAI(self.gameBoard, self.p2Model)

        winner2 = processBoardState(self.gameBoard)

        if winner2 == 2:
            print('LOSS')
            self.losses += 1
            return self.gameBoard, LOSS_REWARD, True, {}
        elif winner2 == 3:
            print('TIE')
            self.ties += 1
            return self.gameBoard, TIE_REWARD, True, {}
    
        return self.gameBoard, VALID_MOVE_REWARD, False, {}
          
    def reset(self):
        self.gameBoard = np.array([0 for space in range(9)])
        return self.gameBoard

    def render(self, mode='human'):
        s = ''
        for i, val in enumerate(self.gameBoard):
            if i % 3 == 0:
                s+='\n'
            s += f' {val} ' 

        print(s)

    def close(self):
        print('Shutting down the environment...')