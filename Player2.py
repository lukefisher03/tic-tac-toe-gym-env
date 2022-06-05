#TODO: implement a 2nd player for the AI to play against -> first random and then with mild skill level
import random

class Player2:
    def __init__(self, agent=0) -> None:
        self.agent = agent
        
    def move(self, board) -> None:
        #print('p2 moving')
        taken = True
        while taken:
            spot = random.randrange(0,9)
            if board[spot] == 0:
                taken = False
                board[spot] = 2
    def moveHuman(self, board):
        #TODO: Implement human player
        taken = True
        while taken:
            spot = int(input('Input desired spot: '))
            if board[spot] == 0:
                taken = False
                board[spot] = 2
    def moveAI(self, board, model):
        action, _state = model.predict(board)
        board[action] = 2
