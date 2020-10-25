import torch
import argparse
import cv2
from tetris_env import Tetris # importing the tetris environment


def get_args():
    '''This function is used for the deep q network to play tetris. Everything is passed to the parser '''
    parser = argparse.ArgumentParser() # creats a parser 
    parser.add_argument("--width", type=int, 
                        default=10, help="The common width for all images")
                        # sets the width of the all images to 10 
    parser.add_argument("--height", type=int, 
                        default=20, help="The common height for all images")
                        # sets the height of the all images to 20 
    parser.add_argument("--block_size", type=int, 
                        default=30, help="Size of a block")
                        # sets the size of the blocks to 30
    parser.add_argument("--fps", type=int, 
                        default=300, help="frames per second")
                        # sets the fps to 300
    parser.add_argument("--saved_path", type=str, 
                        default="trained_models")
                        #sets the saves path to the trained models folder
    parser.add_argument("--result", type=str, 
                        default="after-training.mp4")
                        #outputs the results as a after-training.mp4 video
    args = parser.parse_args() 
    return args

def test(opt):
    '''This function is used for testing the trained model ''' 
    if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
        torch.cuda.manual_seed(125) # the torch cuda manual seed is used in order to have reproducable results
    else:
        torch.manual_seed(125) # sets the random number generator from pytorch
    if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
        testing_model = torch.load("{}/tetris".format(opt.saved_path)) # loads the tetris model from the saved path on the trained models folder
    else:
        testing_model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage) # loads the tetris model from the saved path on the trained models folder the map location is set to lambda and the location is the storage which is the last checkpoint saved on the gpu device
    testing_model.eval() # sets the testing model to evaluation mode
    environment = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size) # sets the environment to the tetris environment that was created before with the width from the parser the height from the parser and the block size from the parser
    environment.reset() # resets the environment
    if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
        testing_modelmodel.cuda() # sets the .cuda() to the testing model to keep track of the gpu
    output_testing_video = cv2.VideoWriter(opt.result, cv2.VideoWriter_fourcc(*'FMP4'), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size)) # here i ouput a video during testing by using the open cv video writer with the result from the parser and i set the format as fmp4 and also i took the fps from the parser and 1.5* width and the block size from the parser.
    while True: 
        next_steps = environment.get_next_states() # next steps are set to the environment next states
        next_actions, next_states = zip(*next_steps.items()) # next actions and next states are zipped together as a tuple
        next_states = torch.stack(next_states) # next states are set to the cocatenates of the next states to a new dimension
        if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
            next_states = next_states.cuda() # sets the .cuda() to the next states to keep track of the gpu
        preds = testing_model(next_states)[:, 0] # the predictios are set to the testing model next states
        idx = torch.argmax(preds).item() #index set the maximum predictions 
        a = next_actions[idx] # a is the next actions index
        _, done = environment.make_step(a, cv2_rend=True, video=output_testing_video) # this is called by itself because of '_,' and done is equal to the environemnt of a, the open cv2 render is true to have a visual look on the game and the video is set to the output testing video
        if done:
            output_testing_video.release() # outpus the testing results as a video when its done
            break # stops
        
if __name__ == "__main__":
    opt = get_args() # the opt is set to the parser arguments
    test(opt) # tests the opt
