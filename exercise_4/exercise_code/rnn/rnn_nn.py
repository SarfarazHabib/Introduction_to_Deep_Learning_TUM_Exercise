import torch
import torch.nn as nn
#from torch.autograd import

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_hh = nn.Linear(self.hidden_size, self.hidden_size, bias = True)
        self.W_xh = nn.Linear(self.input_size, self.hidden_size, bias = True)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unrecognized activation. Allowed activations: tanh or relu")

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        if h == None:
            h = torch.zeros((1, x.size(1), self.hidden_size)).float()

        for t in range(x.size(0)):
            # update the hidden state
            # h = torch.relu(self.W_xh(x[t, :]) + self.W_hh(h))
            h = self.activation(self.W_hh(h) + self.W_xh(x[t]))
            h_seq.append(h)

        h_seq = torch.cat(h_seq, 0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_hh = nn.Linear(self.hidden_size, 4*self.hidden_size, bias = True)
        self.W_xh = nn.Linear(self.input_size, 4*self.hidden_size, bias = True)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
            
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq=[]
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        if h == None:
            h = torch.zeros((1, x.size(1), self.hidden_size)).float()

        if c == None:
            c = torch.zeros((1, x.size(1), self.hidden_size)).float()

        h_seq = []
        for t in range(x.size(0)):
            # update the hidden state
            update_h = self.W_hh(h)
            update_x = self.W_xh(x[t].unsqueeze(0))

            gates = (update_h[:, :, : 3*self.hidden_size] + update_x[:, :, : 3* self.hidden_size]).sigmoid()
            update = (update_h[:, :, 3*self.hidden_size: ] + update_x[:, :, 3* self.hidden_size: ]).tanh()

            fg = gates[:, :, :self.hidden_size]
            ig = gates[:, :, self.hidden_size:2*self.hidden_size]
            og = gates[:, :, 2*self.hidden_size:]

            c = c * fg + update * ig
            h = og * c.tanh()
            h_seq.append(h)

        h_seq = torch.cat(h_seq, 0)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################
        self.RNN = RNN(input_size=input_size,hidden_size = hidden_size)
        self.predict = nn.Linear(hidden_size, classes)
        self.softmax = nn.Softmax(dim=2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        _, h = self.RNN(x)
        y = self.predict(h)
        print(y)
        print(y.size())
        prediction = self.softmax(y)
        print(prediction)
        print(prediction.size())
        return prediction.squeeze(0)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a LSTM classifier                                           #
        ############################################################################
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.predictor = nn.Linear(hidden_size, classes)
        self.softmax = nn.Softmax(dim=2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        _, (h, c) = self.LSTM(x)
        y = self.predictor(h)
        prediction = self.softmax(y)
        return prediction.squeeze(0)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
