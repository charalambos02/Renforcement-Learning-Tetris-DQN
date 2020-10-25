import numpy as np
from PIL import Image
import cv2
import random
import torch
from matplotlib import style
style.use("ggplot")


class Tetris:
    '''Creting the tetris environment and it contains all the function we will need for train and test.
    The tetris environment is created manualy using open cv'''
    #initializing each piece color. I've used the default colors of tetris 
    block_pieces_colors = [(0, 0, 0),(255, 255, 0),(145, 86, 252),(51, 172, 146),(255, 0, 0),(105, 219, 235),(255, 155, 35),(0, 0, 255)]
    #creating each individual pice for the game
    block_pieces = [ [[1, 1],[1, 1]],[[0, 2, 0],[2, 2, 2]],[[0, 3, 3],[3, 3, 0]],[[4, 4, 0],[0, 4, 4]],[[5, 5, 5, 5]],[[0, 0, 6],[6, 6, 6]],[[7, 0, 0],[7, 7, 7]]]

    def __init__(self, height=20, width=10, block_size=20):
        ''' The init function contains the height of the piece, the width of the piece and the block size'''
        self.height = height #initializing the height of the piece
        self.width = width  #initializing the width of the piece
        self.block_size = block_size  #initializing the block size
        #creating a new boeard with the height times the block size and the width divided by the blocksize
        self.second_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([169, 169, 169], dtype=np.uint8) # plus a numpy array of the blocks
        self.color_of_the_text = (255, 255, 153) # seting the color of the text on the board
        self.reset() #resets the environment / board

    def reset(self):
        '''This is the reset function that will be used to reset the environment / board'''
        self.board = [[0] * self.width for _ in range(self.height)] # the board is set to the position 0 * (looping and ignoring the height values 
        self.player_score = 0 # sets the player score on the board to 0 
        self.tetris_blocks = 0 # sets the tetris blocks to 0 
        self.completed_lines = 0 # set the completed lines on the board to 0 
        self.total_block_pieces = list(range(len(self.block_pieces))) # sets the total block pieces to a list of the length of the block pieces
        random.shuffle(self.total_block_pieces) # shuffles randomly all the total block pieces
        self.ind = self.total_block_pieces.pop() # setting the ind to the total number of block pieces and using the .pop() to remove and reutrn the last given index value
        self.piece = [row[:] for row in self.block_pieces[self.ind]] # loops through the block pieces and the ind and sets the piece to the sliece element in the row array 
        self.current_position = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0} # sets the current position to a dictionary with the X as the width //2 - the length of the first piece // by 2 and the Y as 0
        self.end_of_the_game = False # sets the end of the game to false witch means that the game is still running
        return self.get_current_state(self.board) # return the current state of the board

    def rotate_piece(self, piece):
        ''' This function is used to reset the pieces in the environment'''
        number_of_default_rows = number_of_new_columns = len(piece) # sets the default number of rows to the default number of columns equal to the lenght of the piece
        number_of_new_rows = len(piece[0]) # sets the number of new rows the length of the first piece
        reversed_arr = [] # creating a list to store the reversed array
        for i in range(number_of_new_rows): # loops through the number of new rows 
            new_row = [0] * number_of_new_columns # sets the new row to the first position * the number of new columns 
            for j in range(number_of_new_columns): # loops through the number of new columns 
                new_row[j] = piece[(number_of_default_rows - 1) - j][i] # sets the new row to the piece of the number of default rows - 1
            reversed_arr.append(new_row) # append the new rows to the reversed array list 
        return reversed_arr # return the reversed array list

    def get_current_state(self, board):
        '''This function is used to get the current state of the board '''
        lines_cleared, board = self.completed_rows(board) # the cleared lines in the board are equl to the complete rows of the environment board
        # The holes variable is the empty space between each tetris piece on the board 
        holes = self.get_empty_space(board) # the holes are equal to the empty space on the board 
        bumpiness, height = self.get_empty_space_and_bump(board) # bumpiness and the height is set to the empty space of the board

        # torch.FloatTensor returns a multidiamentional matrix
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height]) # return the clearned lines the holes the bumpiness and the height of the board in a multidiamentional matrix 

    def get_empty_space(self, board):
        '''This function is used to get the empty space on the board '''
        number_of_empty_spaces = 0 # number of empty spaces is set to 0 at the begining of the game
        for i in zip(*board): # loops through the board and zips together the board in a tuble
            row = 0 # sets the rows to 0 
            while row < self.height and i[row] == 0: # while the row is less than the height and the row is equl to 0
                row += 1 # adds 1 to the row 
            number_of_empty_spaces += len([x for x in i[row + 1:] if x == 0]) # loops through the rows +1 and if the x in equal to 0 it set the length to the number of empty spaces 
        return number_of_empty_spaces # returns the number of empty spaces 

    def get_next_states(self):
        '''This function is used to get the next states'''
        states = {} # states is set to a dictionary 
        piece_number = self.ind # the piece number is set to the ind
        current_piece = [row[:] for row in self.piece] # loops throught pieces in each row and stores them in the current piece which is the current piece in each row
        if piece_number == 0:  # checks if the piece number is equal to 0
            number_of_piece_rotations = 1 # sets the number of piece rotations to 1
        elif piece_number == 2 or piece_number == 3 or piece_number == 4: # else if the piece number is 2 or the piece number is 3 or the piece number is 4
            number_of_piece_rotations = 2 # sets the number of piece rotations to 2
        else:
            number_of_piece_rotations = 4 # sets the number of piece rotations to 4

        for i in range(number_of_piece_rotations): # loops through the number of piece rotations 
            valX = self.width - len(current_piece[0]) # sets the value of x to the width - the length of the current piece 0
            for x in range(valX + 1): # loops through the range of values X + 1
                piece = [row[:] for row in current_piece] # loops through the current piece in the row and sets the piece to the row position
                pos = {"x": x, "y": 0} # the position of X is set to x and the position of y sets to 0 all together in a dictoniary
                while not self.check_for_pieces_clashes(piece, pos): # infiny loop that checks for piece clashes for each piece and each position
                    pos["y"] += 1 # adds 1 to the position of y
                self.make_rows_shorter(piece, pos) # makes the rows shorted at each piece position
                board = self.store_posXY_in_board(piece, pos) # the board is set to the stored position of X and Y in the board with the piece and position
                states[(x, i)] = self.get_current_state(board) #the states of x and i are set to the current state in the board
            current_piece = self.rotate_piece(current_piece) # the current piece is set to the rotated piece in the current position
        return states # returns the states 

    def get_empty_space_and_bump(self, board):
        ''' This function is used to get the empty spaces and bmpines on the board '''
        board = np.array(board) # the board is set to a numpy array of the board
        jj = board != 0 # the jj variable is set to the board and not 0
        reverse_heights = np.where(jj.any(axis=0), np.argmax(jj, axis=0), self.height) # the reversed heights are set to the  choosen elements at any axis with the indicies of the max values of jj and the height
        heights = self.height - reverse_heights # the heights are set the height - the reversed heights 
        sum_of_heights = np.sum(heights) # the sum of heights is equal to the sum of heights
        current_heights = heights[:-1] # the current heights are set the heights with extended slice -1
        next_heights = heights[1:] # the next heights are set to the heights with extended slice 1
        difference_between_heights = np.abs(current_heights - next_heights) # the difference between heights is set to the absolut value of the current heights - the next heights 
        sumBump = np.sum(difference_between_heights) # the sum of the bumpines is set to the sum of the difference between heights 
        return sumBump, sum_of_heights # returns the sum of the bumpines and the. sum of heights

    def check_for_pieces_clashes(self, piece, pos):
        '''This function is used to check for pieces clashes '''
        new_y = pos["y"] + 1 # the new y is set the position of y + 1 which is the next y 
        for y in range(len(piece)): # loops throught the length of piece
            for x in range(len(piece[y])): # loops throight the length of y piece
                if new_y + y > self.height - 1 or self.board[new_y + y][pos["x"] + x] and piece[y][x]: # if the new y and y and greater that the height -1 or the board of new y + y and the position of x  and the piece of y and x
                    return True # returns true 
        return False    # otherwise returns false 
    
    def new_block_piece(self):
        '''This function is used for the new block pieces '''
        if not len(self.total_block_pieces): # if no the length of total block pieces
            self.total_block_pieces = list(range(len(self.block_pieces))) # sets the total block pieces to a list of all the block pieces
            random.shuffle(self.total_block_pieces) # randomly shuffles the total block pieces (gets a random block piece)
        self.ind = self.total_block_pieces.pop() # setting the ind to the total number of block pieces and using the .pop() to remove and reutrn the last given index value
        self.piece = [row[:] for row in self.block_pieces[self.ind]] # loops through the blovk pieces and sets the row position of ind to the piece 
        self.current_position = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0} # sets the current position to a dictionary with the X as the width // 2 - the length of the piece at position 0 // 2 and the y as 0
        if self.check_for_pieces_clashes(self.piece, self.current_position): # checks if the pieces clases at the current position of the piece
            self.end_of_the_game = True # if the pieces clashes together it ends the game 

    def get_current_board_state(self):
        '''This function is used to get the current board state '''
        board = [x[:] for x in self.board] # loops through the board and sets the board to the position of x 
        for y in range(len(self.piece)): # loops through the length of piece 
            for x in range(len(self.piece[y])): #loops through the length of piece y
                board[y + self.current_position["y"]][x + self.current_position["x"]] = self.piece[y][x] # the board of y and the current position of y and the current position of x + x are equal to the piece y and x
        return board     # retuns the board
    
    def make_rows_shorter(self, piece, pos):
        '''This function is used to make the rows shorter when they are completed '''
        end_of_the_game = False # the end of the game is set to false
        lastClashed_row = -1 # the last clashed row is set to -1
        for y in range(len(piece)): #loops through the range of the piece
            for x in range(len(piece[y])): #loops through the range of the piece y
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]: # if the position y in the board and + y and the position x + x and the piece of y and x then
                    if y > lastClashed_row: # check if y is less than the clashed row
                        lastClashed_row = y # sets the last clashed row to y
        if pos["y"] - (len(piece) - lastClashed_row) < 0 and lastClashed_row > -1: #checks if the position of y - the length of the piece - the last clashed row are greater than 0 and the last clashed row is less than -1 
            while lastClashed_row >= 0 and len(piece) > 1: # while the last clashed row is less and equal than 0 and the length of the piece is less than 1
                end_of_the_game = True # ends the game
                lastClashed_row = -1 # the last clashed row sets to -1
                del piece[0] # delets the piece on the position 0 
                for y in range(len(piece)): # loops through the range of the piece 
                    for x in range(len(piece[y])): # loops through the range of the piece y 
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > lastClashed_row: # fi the position of the y on the board + y and the position of x + x and the piece y and x and if y is less than the last clashed row
                            lastClashed_row = y # sets the last clashed row to y
        return end_of_the_game # ends the game

    def completed_rows(self, board):
        '''This function is used to check for the completed rows on the board ''' 
        del_rows = [] # deleted rows is an empty list 
        for i, row in enumerate(board[::-1]): # loops though the enumerate list of the bord 
            if 0 not in row: # checks if 0 is not in the row
                del_rows.append(len(board) - 1 - i) # appends the length of the board -1 -i in the del rows list 
        if len(del_rows) > 0: # checks if the length of deleted orws is less than 0 
            board = self.remove_completed_rows(board, del_rows) # sets the board to the removed compled rows on the board and the del rows 
        return len(del_rows), board # return the length of the deleted rows and the board
    
    def store_posXY_in_board(self, piece, pos):
        '''This function is used to store the position of X and Y in the board ''' 
        board = [x[:] for x in self.board] # loops throught the board and set the bord to the slice element of x
        for y in range(len(piece)): # loops throught the length of the piece
            for x in range(len(piece[y])): # loops through the length of y
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]: # if the piece y and x and not the position of y+y on the board + the position of x + x
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x] # the position of y and y on the board and the position of x +x on the board are s et to the piece y and x
        return board


    def make_step(self, action, cv2_rend=True, video=None): # has inputs the action, the cv2_rend which is a drawing function in open cv and the video to none
        '''This function is used to make step / play ''' 
        x, number_of_piece_rotations = action # action is set to the x and the number of piece rotations
        self.current_position = {"x": x, "y": 0} # the current position is set to the dictionary of x and y as 0
        for _ in range(number_of_piece_rotations): # loops through the range of the number of piece rotations. in that case we dont care about the number of piece rotations that why  '_' is used 
            self.piece = self.rotate_piece(self.piece) # allows piece rotations
        while not self.check_for_pieces_clashes(self.piece, self.current_position): # while not any pieces are clashed at the current position 
            self.current_position["y"] += 1 # the current position is set to y +1
            if cv2_rend: # if the cv2 render 
                self.cv2_rend(video) # plays the game
        board_full = self.make_rows_shorter(self.piece, self.current_position) # the board full is set to the shorter rows of the current position and the piece 
        if board_full: # if the board is full
            self.end_of_the_game = True # game ends 
        self.board = self.store_posXY_in_board(self.piece, self.current_position) # the position of X and Y at the current position and piece are set to the board
        lines_cleared, self.board = self.completed_rows(self.board) # the cleared lines and the board are set to the completed rows of the board
        player_score = 1 + (lines_cleared ** 2) * self.width # the player score is set to 1 + the cleared lines **2 * the width
        self.player_score += player_score # the player score is passed to the player total score
        self.tetris_blocks += 1 # the tetris block are increased by 1
        self.completed_lines += lines_cleared # the completed lines are equal to the cleared lines
        
        if not self.end_of_the_game: # if the game is not ended
            self.new_block_piece() # gets a new random block piece 
        if self.end_of_the_game: # if the game ends 
            self.player_score -= 2 # the player looses 2 points from his score
        return player_score, self.end_of_the_game # returns the player score and the ends the game

    def remove_completed_rows(self, board, indices):
        ''' This function is used to remove the completed rows on the board '''
        for i in indices[::-1]: # loops through the indicies with slice -1
            del board[i] # deletes the board at the current i position 
            board = [[0 for _ in range(self.width)]] + board # loops through the length of the width and sets the board to 0. The '_' is used because we dont care of the 0
        return board    
    
    def cv2_rend(self, video=None):
        '''This function is used to create the video representation of the tetris game on open cv '''
        if not self.end_of_the_game: # if the game is not ended
            game_image = [self.block_pieces_colors[p] for row in self.get_current_board_state() for p in row] #game game image is set to the block pieces color the current board state and the row
        else:
            game_image = [self.block_pieces_colors[p] for row in self.board for p in row] # loops through the rows on the board and sets the game image to the block pieces colors 
        game_image = np.array(game_image).reshape((self.height, self.width, 3)).astype(np.uint8) # game image is set to a numpy array of the game image and the height and width is reshaped and it converted to uint8
        game_image = game_image[..., ::-1] # the game image is set the the position slieced -1 
        game_image = Image.fromarray(game_image, "RGB") # image. fromarray is used to create an image memore for the game image exprting the array buffer.
        game_image = game_image.resize((self.width * self.block_size, self.height * self.block_size)) # the game image is rezised by the width * the block size and the hight * the block size
        game_image = np.array(game_image) # the game image is stored to a numpy array 
        game_image[[i * self.block_size for i in range(self.height)], :, :] = 0 # loops through the height range and the i* block size is set to 0
        game_image[:, [i * self.block_size for i in range(self.width)], :] = 0 # loops through the width and the i * the block size is set to 0
        game_image = np.concatenate((game_image, self.second_board), axis=1) # the game image and the second board are concantenated togher 
        cv2.putText(game_image, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size), # here we create the player score text on the game board and we take the width * the block size + the block size /2 and the block size
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.
        cv2.putText(game_image, str(self.player_score), # here we add the player score to the score which will show the total score
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size), # we get the width * the block size + the block size /2 and 2 * times the block size 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.
        cv2.putText(game_image, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size), # here we create the pieces text on the game board and we take the width * the block size + the block size / 2 and 4 * block size
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.
        cv2.putText(game_image, str(self.tetris_blocks), # here we add the tetris block to the pieces which will show the total pieces
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size), # we get the width * the block size + the block size /2 and 5 * times the block size 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.

        cv2.putText(game_image, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),  # here we create the lines text on the game board and we take the width * the block size + the block size / 2 and 7 * block size
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.
        cv2.putText(game_image, str(self.completed_lines), # here we add the completed lines to the lines which will show the total completed lines
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size), # we get the width * the block size + the block size /2 and 8 * times the block size 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=self.color_of_the_text) # here the hershey trplex text font is used. There are so many text styles in open cv to choose. The fontscale is set to 1 to fit the board and the color is set as the color of the text.

        if video:
            video.write(game_image) #this is used to record a video for the game play
        cv2.imshow("Tetris with Deep Q-Learning", game_image) # this is the title that is showed to the game board
        cv2.waitKey(1) # here the game waits for 1 second