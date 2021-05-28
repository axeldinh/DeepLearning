import torch
import torch.nn as nn
import torch.nn.functional as F


#######################################
##### MODELS

class Model(nn.Module):
    """
    class Model from which our models will inherit.
    It contains the functions:
        - forward(self, x): returns the outputs of the model
        - loss(self, outputs, target, classes): returns the loss of the model
        - correct_predictions(self, outputs, target, classes): returns the number of correct predictions
    """
    
    def __init__(self, *input):
        super().__init__()
        
    def forward(self, x):
        """
        Returns the outputs of the model. The first output is always the one used for predictions.
        Returns:
            - outputs (tuple(Tensor)): returns one or many outputs from the model (e.g, (output0, output1, output2))
        """
        raise NotImplementedError
        
    def loss(self, outputs, targets, classes):
        """
        Returns the loss from the model given the outputs, targets and classes.
        As the loss function is implemented in the specific model, we can return losses
        either on the digits values, the binary target or both (which is useful for auxiliary losses).
        
        Returns:
            - loss (float): loss used to optimize the model
        """
        raise NotImplementedError
        
    def correct_predictions(self, outputs, targets, classes):
        """
        Returns the number of correct predictions given the outputs, target and classes.
        As the function is implemented in the specific model, we can either return the
        predictions on the digits values or the binary target.
        
        Returns:
            - predictions (tuple(int)): tuple of predictions, the first element is the main prediction of the model
        """
        raise NotImplementedError

#######################################
#### Multi-Layer Perceptrons
#### Here we implement the MLPBlock Module and the MLP Model

class MLPBlock(nn.Module):
    """
    Returns an MLP with the following structure: Linear -> ReLU -> Dropout -> Linear -> ...
                                                 Dropout -> Linear
    Note that Dropout and ReLU are not used after the last layer.
    
    Parameters:
        - layers_sizes (list(int)): list of features sizes, e.g [2, 5, 10] will return Linear(2,5) -> ReLU -> Linear(5,10),
                                    so len(layers_sizes) = number of desired layers + 1
        - dropouts (list(int))    : list of the dropouts probabilities across the layers, e.g [0.5] will return
                                    Linear -> ReLU -> Dropout(0.5) -> Linear. 
                                    Note that we do not use dropout after the last layer,
                                    so len(dropouts) = number of desired layers - 1
    
    Returns:
        - MLPBlock (nn.Module): Module with the same structure as mentioned above, e.g
                                MLPBlock([100, 50, 10], [0.5]) will return a Module with:
                                Linear(100, 50) -> ReLU() -> Dropout(0.1) -> Linear(50, 10)
    """

    def __init__(self, layers_sizes = [392, 200, 100, 2], dropouts = [0.0, 0.0]):

        super().__init__()
        
        self.activation = nn.ReLU()

        self.fcs = nn.Sequential(
            *(nn.Sequential(
              nn.Linear(layers_sizes[i], layers_sizes[i+1]),
              self.activation,
              nn.Dropout(dropouts[i])) for i, _ in enumerate(layers_sizes[:-2]))
        )

        self.fcout = nn.Linear(layers_sizes[-2], layers_sizes[-1])

    def forward(self, x):

        return  self.fcout(self.fcs(x.view(x.size(0), -1)))
        
        
        
class MLP(Model):
    """
    Returns a Model with an MLP structure and a CrossEntropyLoss.
    See the MLPBlock function for the structure of the MLP.
    The MLP Model can be used for digit recognition or digits comparisons with the digit argument.
    IMPORTANT: if the input of the forward function contains more than 1 digit (for one sample), 
               the model will train only on the first one. Allowing the training of digit recognition models with the same dataset.
               
    Parameters:
        - layers_sizes (list(int)): list of features sizes, e.g [2, 5, 10] will return Linear(2,5) -> ReLU -> Linear(5,10),
                                    so len(layers_sizes) = number of desired layers - 1
        - dropouts (list(int))    : list of the dropouts probabilities across the layers, e.g [0.5] will return
                                    Linear -> ReLU -> Dropout(0.5) -> Linear. 
                                    Note that we do not use dropout after the last layer,
                                    so len(dropouts) = number of desired layers - 2
        - digit (bool)            : Whether or not the model should recognize a digit, if True the losses and predictions
                                    will be done on the classes rather than the targets
    
    Returns:
        - MLP (Model): An MLP with CrossEntropyLoss() as criterion
    """

    def __init__(self, layers_sizes, dropouts, digit = False):
        
        super().__init__()

        self.digit = digit
        self.mlp = MLPBlock(layers_sizes, dropouts)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        
        # If digit is True check if the input contains only 1 digit
        # otherwise take the 1rst one
        if self.digit and x.size(1) > 1:
            x = x[:, 0, :, :].unsqueeze(1)

        return [self.mlp(x.view(x.size(0), -1))]

    def loss(self, outputs, targets, classes):
        
        # If digit is True check if the input contains only 1 digit
        # otherwise take the classes from the 1rst one
        if self.digit:
            if classes.size(1) > 1:
                classes = classes[:,0]

            return self.criterion(outputs[0], classes)
        else:
            return self.criterion(outputs[0], targets)

    def correct_predictions(self, outputs, targets, classes):

        _, predictions = torch.max(outputs[0], dim = -1)
        
        # If digit is True check if the input contains only 1 digit
        # otherwise take the classes from the 1rst one
        if self.digit:
            if classes.size(1) > 1:
                classes = classes[:,0]
                
            return [torch.sum(predictions == classes)]
        
        else :
            return [torch.sum(predictions == targets)]
            

#######################################
#### Convolutional Neural Networks
#### Here we implement the ConvBlock Module and the ConvNet Model

class ConvBlock(nn.Module):
    """
    Returns a Convolutional Block with the following structure: 
        Conv2d(in_channels, out_channels) -> BatchNorm2d(out_channels) -> ReLU() -> MaxPool(2).
    
    Parameters:
        - in_channels (int): Number of channels to enter the Conv2d layer
        - out_channels (int): Number of channels at the end of the Conv2d layer
        - kernel_size (int): kernel size for the Conv2d layer
        
    Returns:
        - ConvBlock (Module): Module with the same structure as mentioned above.
    """

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()

    def forward(self, x):

      return self.pool(self.activation(self.bn(self.conv(x))))


class ConvNet(Model):
    """
    IMPORTANT: For convenience, the kernel size is fixed, this way we can easily predict
               the images sizes at the end of the ConvBlocks. But we must give 2 ConvBlocks
               for it to work, otherwise the Model will not run.
               
    Returns a ConvNet Model with a sequence of ConvBlock Modules followed by
    a small MLPBlock for the classification, and a CrossEntropyLoss.
    Note that the kernel sizes here is always 3.
        
    See the ConvBlock function for the structure of the convolutional layers.
    See the MLPBlock function for the structure of the classification layer.
        
    The ConvNet Model can be used for digit recognition or digits comparisons with the digit argument.
    
    Parameters:
        - channels (list(int))  : list of the channels to use for the ConvBlock's.
                                  Note that len(channels) = number of desired ConvBlocks + 1
        - hidden (int)          : number of units used for the classification's hidden layer.
        - digit (bool)          : Whether or not the model should recognize a digit, if True the losses and predictions
                                  will be done on the classes rather than the targets
                                  
    Returns:
        - ConvNet (Model): A ConvNet with appropriate loss and forward pass.
                           Example: ConvNet([2, 32, 64], 50, False, False) will return:
                                    Conv2d(2, 32, 3) -> BatchNorm2d(32) -> ReLU() -> MaxPool2d(2)
                                    -> Conv2d(32, 64, 3) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(2)
                                    -> Linear(4*64, 50) -> ReLU() -> Linear(50, 2)
    """

    def __init__(self, channels, hidden, digit = False):

        super().__init__()

        self.digit = digit
        self.criterion = nn.CrossEntropyLoss()
        
        out_features = 10 if self.digit else 2

        self.convs = nn.Sequential(
            *(ConvBlock(channels[i], channels[i+1], 3) for i, _ in enumerate(channels[:-1]))
        )
        
        in_features = channels[-1]*4 # Get the number of features after self.convs
        
        self.classifier = MLPBlock(layers_sizes = [in_features, hidden, out_features], dropouts = [0.0])

    def forward(self, x):
        
        # If self.digit, verify that the input contains only one digit
        # otherwise keep the 1rst one.
        if (self.digit) and (x.size(1) > 1):
            x = x[:,0,:,:].unsqueeze(1)
            
        x = self.convs(x)

        return [self.classifier(x.view(x.size(0), -1))]

    def loss(self, outputs, targets, classes):
        
        # If digit is True check if the input contains only 1 digit
        # otherwise take the classes from the 1rst one
        if self.digit:
            if classes.size(1) > 1:
                classes  = classes[:,0]

            return self.criterion(outputs[0], classes)

        else:
            return self.criterion(outputs[0], targets)

    def correct_predictions(self, outputs, targets, classes):

        _, predictions = torch.max(outputs[0], dim = -1)
        
        # If digit is True check if the input contains only 1 digit
        # otherwise take the classes from the 1rst one
        if self.digit:
            if classes.size(1) > 1:
                classes = classes[:,0]

            return [torch.sum(predictions == classes)]

        else:
            return [torch.sum(predictions == targets)]
            
            
#######################################
#### Auxiliary Model
#### Here we implement the AuxModel Model


class AuxModel(Model):
    """
    Returns a AuxModel, it takes a Model handle and a config to permit different initialization of the same model.
    Here the model defined by Model_fn(*config) is used to make digit predictions, then a small MLP is used to make
    the final prediction. The predictions are made using a CrossEntropyLoss.
    The final loss is computed with the following relation:
        CrossEntropyLoss(targets_predictions) + aux_coeff * (CrossEntropyLoss(digit1_predictions) + CrossEntropyLoss(digit2_predictions))
    
    Parameters:
        - Model_fn (Model handle): handle for the digit recognizer Model
        - config (list): list of the arguments necessary for the call to Model(*config)
        - weight_sharing (bool): if True, the digits recognition will be done on the same model, else we will create
                                 two distinct models from the same function handle.
        - aux_coeff (float): Tells how much the loss on the digits predictions accounts for the final loss
        
    Returns:
        - AuxModel (Model): Returns a siamese fashion model which is trained on digit recognition
                            before the final classification.
                            Example:
                                For AuxModel(MLP, config, False, False), we get:
                                MLP1(digit1) --\
                                                concat -> Linear(20,100) -> ReLU() -> Linear(100, 2) -> output
                                MLP2(digit2) --/
    """

    def __init__(self, Model_fn, config, weight_sharing, aux_coeff):
        
        super().__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.weight_sharing = weight_sharing
        self.aux_coeff = aux_coeff
        
        # if weight_sharing is True we only create 1 model
        # else we create 2 separated models
        if weight_sharing:
            self.imgnet1 = Model_fn(*config)
        else:
            self.imgnet1 = Model_fn(*config)
            self.imgnet2 = Model_fn(*config)
            
        # We create a MLPBlock to combine the previous results
        # and get the final output
        self.mlpout = MLPBlock([20, 100, 2], [0.0])

    def forward(self, x):
        
        # Get the 2 images
        img1 = x[:, 0, :, :].unsqueeze(1)
        img2 = x[:, 1, :, :].unsqueeze(1)
        
        # Get the logits for digit predictions
        if self.weight_sharing:
            pred_digit1 = self.imgnet1(img1)
            pred_digit2 = self.imgnet1(img2)
        else:
            pred_digit1 = self.imgnet1(img1)
            pred_digit2 = self.imgnet2(img2)
        
        # Concatenation of the outputs
        x = torch.cat([pred_digit1[0], pred_digit2[0]], dim = -1).view(x.size(0), -1)
        
        # Final layers for the binary classification
        x = self.mlpout(x)

        return (x, pred_digit1[0], pred_digit2[0])

    def loss(self, outputs, targets, classes):

        main_loss = self.criterion(outputs[0], targets)
        aux_loss = self.criterion(outputs[1], classes[:, 0]) + self.criterion(outputs[2], classes[:, 1])

        return main_loss + self.aux_coeff*aux_loss
        
    def correct_predictions(self, outputs, targets, classes):
        
        _, predictions_targets = torch.max(outputs[0], dim = -1)
        _, predictions_digit1 = torch.max(outputs[1], dim = -1)
        _, predictions_digit2 = torch.max(outputs[2], dim = -1)
        
        return (
            torch.sum(predictions_targets == targets),
            torch.sum(predictions_digit1 == classes[:,0]),
            torch.sum(predictions_digit2 == classes[:,1])
            )
        

        