import torch # Only library we are allowed to use

class Model_1(nn.Module):
    def __init__(self, layers):
        super(Model, self)
        self.metrics = Metrics()
        self.conv = nn.Linear(2*14*14, 2)

    def forward(self, x):
        return torch.sigmoid(conv(x))

class Model_2(nn.Module):
    def __init__(self, layers):
        super(Model, self)
        self.metrics = Metrics()
    def forward

class Metrics():
    "Class containing relevant metrics for the model, it updates in the training functions"

    accuracy = [[],
                [],..]
    train_losses = [[],
                    [],..]
    test_losses = [[],
                   [],...]

model.metrics = Metrics()

def full_train(model,.., num_trains):
    "Trains for num_trains random initialization (get a standard deviation)"


def train(model, criterion, optimizer, scheduler, epochs):
    "Trains on 25 epochs the model with given criterion and optimizer"

    for epochs in range(epochs):

def evaluate(model, test_set):
    "returns accuracy on test set"

    return accuracy

