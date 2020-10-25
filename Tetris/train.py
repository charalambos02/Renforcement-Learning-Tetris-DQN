import numpy as np
import argparse
import os
import cv2
import shutil # is used to copy the file destination 
from random import random, randint, sample
import torch
import torch.nn as nn
from collections import deque 
from tensorboardX import SummaryWriter
from deep_q_network import DeepQNetwork # importing the deep q network that i've created before 
from tetris_env import Tetris # importing the tetris environment that i have ccreated bofore 
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def pars_arguments():
    '''This function is used for the deep q network to play tetris. Everything is passed to the parser '''
    parser = argparse.ArgumentParser() # create a new parser
    parser.add_argument("--width", type=int, default=10, 
                       help="Width for all images")
                        # sets the width of the all images to 10 
    parser.add_argument("--height", type=int, default=20, 
                        help="Height for all images")
                        # sets the height of the all images to 20
    parser.add_argument("--block_size", type=int, default=30, 
                        help="Size of a block")
                        # sets the size of the blocks to 30
    parser.add_argument("--mini_batch_size", type=int, default=512, 
                        help="Number of images per batch")
                        # sets the number of images per mini batch to 512
    parser.add_argument("--lr", type=float, 
                        default=1e-3) # sets the learning rate to 1e-3
    parser.add_argument("--gamma", type=float, 
                        default=0.99) # sets the gamma to 0.99
    parser.add_argument("--initialEpsilon", type=float, 
                        default=1) # sets the epsilon to 1
    parser.add_argument("--finalEpsilon", type=float, 
                        default=1e-3) # sets the final epsilon to 1e-3
    parser.add_argument("--decay_epochs", type=float, 
                        default=2200)  # sets the number of decay epochs to 2000
    parser.add_argument("--num_epochs", type=int, 
                        default=2200) # sets the number of epochs for training to 2500
    parser.add_argument("--store_interval", type=int, 
                        default=1000) # sets the store interval to 1000
    parser.add_argument("--mem_size", type=int, default=30000,
                        help="Number of epoches during testing phases") # sets the memory size size to 30000                      
    parser.add_argument("--log_path", type=str, 
                        default="tensorboard") # sets the log path to the tenorboard to show the training stats
    parser.add_argument("--saved_path", type=str, 
                        default="trained_models") # sets the saved path to the trained models
    parser.add_argument("--fps", type=int, 
                        default=300, help="frames per second")
                        # sets the fps to 300
    parser.add_argument("--result", type=str, 
                        default="before_during_training.mp4")
                        #outputs the results as a played_gam.mp4
    args = parser.parse_args()
    return args

# Pytorch helper plotting function
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

        
# Pytorch helper plotting function
tot_reward = []
def plot_reward():
    plt.figure(2)
    plt.clf()
    reward_t = torch.tensor(tot_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        
        
def train(opt):
    '''This function is for the training '''
    if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
        torch.cuda.manual_seed(125) # the torch cuda manual seed is used in order to have reproducable results 
    else:
        torch.manual_seed(125)  # sets the random number generator from pytorch
    if os.path.isdir(opt.log_path): # check if the path is the path that is stored
        shutil.rmtree(opt.log_path) # delets all the content from the lo_path dirfectory
    os.makedirs(opt.log_path) # create a new path directory and store it to the log_path
    new_writer2 = SummaryWriter(opt.log_path) # create a new summary writer with the log path
    environment = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size) # sets the environment to the tetris environment that i have created before with with width, the height and the the block size  from the parser.
    deepQ_model = DeepQNetwork() # the model is set to the deep q network that was created before 
    my_optim = torch.optim.Adam(deepQ_model.parameters(), lr=opt.lr) # sets the optimizaer with the algorith Adamn and the deep q model paramets and the learning rate from the parser
    cn = nn.MSELoss() # this is the default as ((input-target)**2).mean() but with pytorch it gets easier 
    state = environment.reset() # the state is equal to a new reset environment 
    if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
        deepQ_model.cuda() # sets the .cuda() to the deep q learning model to keep track of the gpu
        state = state.cuda() # sets the .cuda() to the state to keep track of the gpu
    r_memory = deque(maxlen=opt.mem_size) #adds the removed elements to the r_memory. In that case the removed element is the memory sizy from the parser
    epoch = 0 #the epoch is set to 0
    output_training_video = cv2.VideoWriter(opt.result, cv2.VideoWriter_fourcc(*'FMP4'), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while epoch < opt.num_epochs: # loops until the epoch is less than the number of epochs from the parser
        
        next_steps = environment.get_next_states() # the next steps are set to the environment next states
        epsilon = opt.finalEpsilon + (max(opt.decay_epochs - epoch, 0) * (opt.initialEpsilon - opt.finalEpsilon) / opt.decay_epochs) # this is for exploration. The epsilon is the final epsilon value from the parser + the max decay epochs - epoch and 0 * with the initial epsilon from the parser - the final epsilon / by the number of decay epochs.
        pp = random() # pp is a random
        rand_action = pp <= epsilon # random action is equal to the pp less than the epsilon
        nextActions, next_states = zip(*next_steps.items()) # next action and next states are equal to a series of tuples of the next steps 
        next_states = torch.stack(next_states) # next states are set to the cocatenates of the next states to a new dimension
        if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
            next_states = next_states.cuda() # sets the .cuda() to the next states to keep track of the gpu
        deepQ_model.eval() # this pytorch function sets the model to evaluation mode while testing
        with torch.no_grad(): # torch. no_grad() is used to deactive the autograd egnine which reduces memory usage and speed up
            dqm_p = deepQ_model(next_states)[:, 0] # press is set to the deepq model with the next states 
        deepQ_model.train() # trains the deep q model
        if rand_action: # if the action is random
            idx = randint(0, len(next_steps) - 1) # the index is set to the random of the length of the next steps -1
        else:
            idx = torch.argmax(dqm_p).item() #index set the maximum values of dqm_p
        next_state = next_states[idx, :] # the next state is equal to the next states index
        action = nextActions[idx] #action is set the next actions index
        reward, done = environment.make_step(action, cv2_rend=True) # the reword and done is set to the environment with the action and the open cv render which is the environment for visualization
        if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
            next_state = next_state.cuda() # sets the .cuda() to the next state to keep track of the gpu
        r_memory.append([state, reward, next_state, done]) # appends the r memory with the state reward next state and done
        if done: # if its done
            output_training_video.release()
            episode_durations.append(epoch + 1)
            #plot_durations()
            final_total_score = environment.player_score # the final total score is equal to the environments players score
            tot_reward.append(final_total_score)
            plot_reward()
            final_total_blocks = environment.tetris_blocks # the final total blocks are equal to the environments tetris blocks
            final_total_completed_lines = environment.completed_lines # the final total completed lines are equal to the environments completed lines
            state = environment.reset() # state is equal to a new environment (rest)
            if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
                state = state.cuda() # sets the .cuda() to the state to keep track of the gpu
        else:
            state = next_state # the state is equal to the next state
            continue
        if len(r_memory) < opt.mem_size / 10: # if the length of the r memory is less than the parsers memory size / 10
            continue # continues 
        epoch += 1 # increments epoch +1
        batch = sample(r_memory, min(len(r_memory), opt.mini_batch_size)) # the batch is set to the sample of the r memory the minimum length of the r memory and the mini batch size from the parser
        stateBatch, batchReward, nextB_state, completed_batch = zip(*batch) # the statebatch, the batch reward the next state and the completed batch are all zipped all together to a tuple 
        stateBatch = torch.stack(tuple(state for state in stateBatch)) # the state batch is equal to the  to the cocatenates as a tuple of the states
        batchReward = torch.from_numpy(np.array(batchReward, dtype=np.float32)[:, None]) # the batch reward is equal to a numpy ndarray of the batch reward as a float
        nextB_state = torch.stack(tuple(state for state in nextB_state)) # the nextB state is equal to the cocatenates as a tuple of the states 
        if torch.cuda.is_available(): # torch.cuda is used for computational perpose and the .availiable() shows if the system supports cuda
            stateBatch = stateBatch.cuda() # sets the .cuda() to the state batch to keep track of the gpu
            batchReward = batchReward.cuda() # sets the .cuda() to the batch reward to keep track of the gpu
            nextB_state = nextB_state.cuda() # sets the .cuda() to the nextB state to keep track of the gpu
        q_values = deepQ_model(stateBatch) # the q values are equal to the models's state batch
        deepQ_model.eval() # sets the model to evaluation mode for testing
        with torch.no_grad(): # torch. no_grad() is used to deactive the autograd egnine which reduces memory usage and speed up
            nextPred_batch = deepQ_model(nextB_state) # the next predi batch is equal to the models's nextB state
        deepQ_model.train() # sets the model to training mode 
        batch_Y = torch.cat(tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in zip(batchReward, completed_batch, nextPred_batch)))[:, None] #  Loops in the zip tuple of batch rewards completed batches and next pred batch and if the batch of Y is equal to a oncatenated tuple of the reward. If its not done the reward + the gamma from the parser * the predictions are stored to the batch Y. 
        my_optim.zero_grad() # the gradients of the optimizer are set to zero at the begining of the mini batch
        loss = cn(q_values, batch_Y) # the loss is equal to the q values and the batch y
        loss.backward() # computes dloss/dx for every parameter x which has requires the grad = True
        my_optim.step() #performs a parameter update on the optimzier based on the current gradient
        print("Epoch Num: {}/{}, Action: {}, Score: {}, TPieces {}, Cleared lines: {}".format(epoch,opt.num_epochs,action, 
            final_total_score,
            final_total_blocks,
            final_total_completed_lines)) # prints the epoch number the action the final total score the final total blocks and the final completed lines for every epoch during training
        new_writer2.add_scalar('Train/Score', final_total_score, epoch - 1) # creates a summury scaler using tensorflow for the train score which gets the final total score and the step which is epoch -1
        new_writer2.add_scalar('Train/TPieces', final_total_blocks, epoch - 1) # creates a summury scaler using tensorflow for the train TPieces which gets the final total blocks and the step which is epoch -1
        new_writer2.add_scalar('Train/Cleared lines', final_total_completed_lines, epoch - 1) # creates a summury scaler using tensorflow for the train cleared lines which gets the final total completed lines and the step which is epoch -1
        if epoch > 0 and epoch % opt.store_interval == 0: # if the epoch is greater than 0 and the epoch % the stored interval is equal to 0
            torch.save(deepQ_model, "{}/tetris_{}".format(opt.saved_path, epoch)) # the trained model and epochsis saved to the saved path which is the trained models folder.
    torch.save(deepQ_model, "{}/tetris".format(opt.saved_path)) # saves the trained model to the saved path from the parser which is the trained models folder
if __name__ == "__main__":
    opt = pars_arguments() #the opt is set to the parsers arguments 
    train(opt) # starts training