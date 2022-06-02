from tabnanny import check
import gym
from gym import spaces
import numpy as np


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
INVALID_MOVE_REWARD = -10
VALID_MOVE_REWARD = 1
WIN_REWARD = 5
LOSS_REWARD = -5
TIE_REWARD = -1

#Returns whether or not someone has won the game. Returns an int with 0 being non-winning state, 1: x wins, 2: o wins, 3: tie.
def checkBoardState(board) -> int:
    for p in [1,2]:
        for row in board:
            if row == [p,p,p]:
                #print(f'p{p} wins by horizontal row')
                return p

        for i in range(3):
            col = []
            for row in board:
                col.append(row[i])
            if col == [p,p,p]:
                #print(f'p{p} wins by vertical column')
                return p

        i = 0
        diag = []
        for row in board:
            diag.append(row[i])
            i+=1
        if diag == [p,p,p]:
            #print(f'p{p} wins by diagonal (top right to bottom left)')
            return p

        i = 2
        diag = []
        for row in board:
            diag.append(row[i])
            i -= 1
        if diag == [p,p,p]:
            #print(f'p{p} wins by diagonal (bottom left to top right)')
            return p

    tie = [row.count(0) for row in board]
    if sum(tie) == 0:#if the number of 0's on the board is 0 then the game is a tie becuase there are no more spots to fill.
        return 3
    return 0

class TicTacToe(gym.Env):
    metadata = {
        'render.modes' : ['human']
    }

    def __init__(self, player2='') -> None:
        super(TicTacToe, self).__init__()

        self.action_space = spaces.MultiDiscrete([3,3])
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,3), dtype=np.int64)
        #initalize an empty game board on each new episode.
        self.gameBoard = [[0,0,0] for col in range(3)]
        self.p2 = player2
        self.wins = 0
        self.invalidMoves = 0
        self.ties = 0
        self.losses = 0

        '''
        side note: Initally the gameBoard would be declared in the global space along with the algorithm. But in python
        class attributes and variables are accessible everywhere. This makes the implementation of a two player game easier,
        the agent will take its turn, and then a second agent will take its turn switching back and forth modifying the gameBoard
        instantiated with each episode.   
        '''
    def step(self, action): 
        #For sake of simplicity, our agent always plays as x's. Whoever goes first will be determined randomly before the start of the game.
        done = False
        reward = VALID_MOVE_REWARD #give a +1 reward for valid actions
        x,y = action
        if self.gameBoard[x][y] != 0:#end the game early and pass extremely negative reward for placing a pieces that's already taken. (cheating)
            done = True 
            reward = INVALID_MOVE_REWARD
            print('Invalid Move')
            self.invalidMoves += 1
            return self.gameBoard, reward, done, {} #make sure to end the episode early so as to not waste training time on unhelpful regions.

        self.gameBoard[x][y] = 1 #Place the agent's piece on the board.
        
        winState = checkBoardState(self.gameBoard)
        if winState == 1:
            reward = WIN_REWARD
            done = True
            print('Win')
            self.wins += 1
            return self.gameBoard, reward, done, {}
        elif winState == 3:
            reward = TIE_REWARD
            done = True
            self.ties += 1
            print('Tie')
            return self.gameBoard, reward, done, {}
        
        p2WinState = self.p2Move()
        reward = p2WinState
        if p2WinState != 1:
            done = True
        return self.gameBoard, reward, done, {}
    def p2Move(self) -> int:
        self.p2.move(self.gameBoard)
        winState = checkBoardState(self.gameBoard)
        if winState == 2:
            reward = LOSS_REWARD
            self.losses += 1
            print('Loss')

        elif winState == 3:
            reward = TIE_REWARD
            print('Tie')
            self.ties += 1
        else:
            reward = VALID_MOVE_REWARD
        return reward
    def reset(self):
        self.gameBoard = [[0,0,0] for col in range(3)]
        return self.gameBoard

    def render(self, mode='human'):
        print(self.gameBoard)
        print('\n\n')

    def close(self):
        print('Shutting down the environment...')