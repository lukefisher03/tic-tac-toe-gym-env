#TODO: implement a 2nd player for the AI to play against -> first random and then with mild skill level
import random

class Player2:
    def move(self, board) -> None:
        #print('p2 moving')
        taken = True
        while taken:
            x = random.randrange(0,3)
            y = random.randrange(0,3)
            if board[x][y] == 0:
                taken = False
                board[x][y] = 2
    def moveHuman(self, board):
        #TODO: Implement human player
        taken = True
        while taken:
            coors = input('Coordinates(x,y)').split(',')

