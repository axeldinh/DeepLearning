from python_scripts import *

# All experiences are run 25 times with 25 epochs and batch size of 100
mini_batch_size = 100
epochs = 50
num_trains = 1

print(f"All runs are done {num_trains} times with {epochs} epochs\n")

final_accuracies = []
best_accuracies = []

#######################
# 1-Run a MLP and a ConvNet on the full data
#   Quick first view on weight sharing

# First get a mean accuracy and standard deviation for the MLP

print("#######################################################")
print("Part 1 - Training of a MLP and a ConvNet trained on the targets without any weight-sharing or auxiliary losses:\n")

# Training of the MLP
config = [ [392, 200, 100, 2], [0.0, 0.0], False] # [ layers_sizes, dropouts, digit ]
model = MLP(*config)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
num_params = compute_number_parameters(model)
print("Training a MLP with {} parameters...".format(num_params))
_, _, accuracies = train(model, optimizer, mini_batch_size, epochs)
print("Training over...")

print("Final accuracy = {:.2f}%, Best Accuracy = {:.2f}%\n".format(accuracies[-1]*100, max(accuracies)*100))

# Training of the ConvNet
config = [ [2, 128, 32], 400, False] # [ channels, hidden, digit ]
model = ConvNet(*config)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
num_params = compute_number_parameters(model)
print("Training a ConvNet with {} parameters...".format(num_params))
_, _, accuracies = train(model, optimizer, mini_batch_size, epochs)
print("Training over...")

print("Final accuracy = {:.2f}%, Best Accuracy = {:.2f}%\n".format(accuracies[-1]*100, max(accuracies)*100))

#######################
# 2-Run a MLP and a ConvNet on 1 digit, in a Siamese fashion, with a small MLP for the decision making
#   We either declare 2 MLP/ConvNet for each digit or 1 for both to see the effect of weight-sharing

print("#######################################################")
print("Part 2 - Training of a MLP and a ConvNet trained on the AuxModel (the Siamese model) with/without weight-sharing, but no auxiliary losses:\n")

# Training of a MLP in AuxModel:

for weight_sharing in [False, True]:
    
    mlp_config = [ [196, 150, 100, 10], [0.0, 0.0], True] # [layers_sizes, dropouts, digit]
    config = [MLP, mlp_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
    model = AuxModel(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a MLP in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
    _, _, accuracies = train(model, optimizer, mini_batch_size, epochs)
    print("Training over...\n")
    print("Final accuracy = {:.2f}%, Best Accuracy = {:.2f}%\n".format(accuracies[-1]*100, max(accuracies)*100))


for weight_sharing in [False, True]:
    
    convnet_config = [ [1, 32, 64], 100, True] # [ channels, hidden, digit ]
    config = [ConvNet, convnet_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
    model = AuxModel(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a ConvNet in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
    _, _, accuracies = train(model, optimizer, mini_batch_size, epochs)
    print("Training over...\n")
    print("Final accuracy = {:.2f}%, Best Accuracy = {:.2f}%\n".format(accuracies[-1]*100, max(accuracies)*100))
    
#######################
# 3-Run of the Siamese ConvNet from part 2 with different auxiliary coefficients
#   The bigger the coefficient, the more the digits' predictions are taken into account

print("#######################################################")
print("Part 3 - Training of the Siamese ConvNet with Weight-Sharing from part 2 with different auxiliary coefficients.")
print("         The bigger the coefficient, the more the digits' predictions are taken into account:\n")

print("For the test.py file, we will only run our best model (with an auxiliary coefficient of 100)...\n")

best_coef = 100
convnet_config = [ [1, 128, 32], 50, True] # [ channels, hidden, digit ]
config = [ConvNet, convnet_config, weight_sharing, best_coef] # [Model_fn, config, weight_sharing, aux_coeff]
model = AuxModel(*config)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
num_params = compute_number_parameters(model)
print("Training a ConvNet in a Siamese fashion with {} parameters, with Weight-Sharing and an auxiliary coefficient of {}...".format(num_params, best_coef))
_, _, accuracies = train(model, optimizer, mini_batch_size, epochs)
print("Training over...\n")

print("Final accuracy = {:.2f}%, Best Accuracy = {:.2f}%\n".format(accuracies[-1]*100, max(accuracies)*100))