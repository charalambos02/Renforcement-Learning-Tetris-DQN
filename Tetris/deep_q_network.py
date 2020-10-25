import torch.nn as nn
'''Importing the pytorch library and the nn which is a great package from neural networks '''

class DeepQNetwork(nn.Module):
    ''' This is the structure of the deep q network'''
    def __init__(self):
        super(DeepQNetwork, self).__init__() # The models should sublass the class
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True)) # first layer with relu activation 
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True)) # second layer with relu activation
        self.conv3 = nn.Sequential(nn.Linear(64, 1)) # third layer
        self._create_weights()

#basic functions of pytorch that were used 
    def _create_weights(self):
        '''This is a function that creates the weights '''
        for i in self.modules(): # loops through all the modules of the init function
            if isinstance(i, nn.Linear): # checks if the instance is linear
                nn.init.xavier_uniform_(i.weight) #sets the xavier unifor and weight
                nn.init.constant_(i.bias, 0) # sets the bias 

    def forward(self, f):
        '''Sequence of layers and processes , call to determine the next action or batch'''
        f = self.conv1(f) # set the first layer to the forward 'f'
        f = self.conv2(f)  # set the second layer to the forward 'f'
        f = self.conv3(f)  # set the third layer to the forward 'f'
        return f
