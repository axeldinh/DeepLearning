from python_scripts import *

# All experiences are run 25 times with 25 epochs and batch size of 100
mini_batch_size = 100
epochs = 50
num_trains = 15

print(f"All runs are done {num_trains} times with {epochs} epochs\n")

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
mean, std, _ = get_mean_std(MLP, config, optimizer, mini_batch_size, num_trains, epochs = epochs)
print("Training over...")

print("\nMean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))

# Training of the ConvNet
config = [ [2, 128, 32], 400, False] # [ channels, hidden, digit ]
model = ConvNet(*config)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
num_params = compute_number_parameters(model)
print("Training a ConvNet with {} parameters...".format(num_params))
mean, std, _ = get_mean_std(ConvNet, config, optimizer, mini_batch_size, num_trains, epochs = epochs)
print("Training over...")

print("\nMean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))

#######################
# 2-Run a MLP and a ConvNet on 1 digit, in a Siamese fashion, with a small MLP for the decision making
#   We either declare 2 MLP/ConvNet for each digit or 1 for both to see the effect of weight-sharing

print("#######################################################")
print("Part 2 - Training of a MLP and a ConvNet trained on the AuxModel with/without weight-sharing, but no auxiliary losses:\n")

# Training of a MLP in AuxModel:

means = []
stds = []
for weight_sharing in [False, True]:
    
    mlp_config = [ [196, 150, 100, 10], [0.0, 0.0], True] # [layers_sizes, dropouts, digit]
    config = [MLP, mlp_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
    model = AuxModel(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a MLP in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
    mean, std, _ = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs)
    print("Training over...\n")
    means.append(mean)
    stds.append(std)

print("For weight_sharing = False: Mean best test accuracy = {:.2f}%(+-{:.2f})".format(means[0]*100, stds[0]*100))
print("For weight_sharing = True : Mean best test accuracy = {:.2f}%(+-{:.2f})\n".format(means[1]*100, stds[1]*100))

means = []
stds = []
for weight_sharing in [False, True]:
    
    convnet_config = [ [1, 32, 64], 100, True] # [ channels, hidden, digit ]
    config = [ConvNet, convnet_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
    model = AuxModel(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a ConvNet in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
    mean, std, _ = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs)
    print("Training over...\n")
    means.append(mean)
    stds.append(std)

print("For weight_sharing = False: Mean best test accuracy = {:.2f}%(+-{:.2f})".format(means[0]*100, stds[0]*100))
print("For weight_sharing = True : Mean best test accuracy = {:.2f}%(+-{:.2f})\n".format(means[1]*100, stds[1]*100))
    
#######################
# 3-Run of the Siamese ConvNet from part 2 with different auxiliary coefficients
#   The bigger the coefficient, the more the digits' predictions are taken into account

print("#######################################################")
print("Part 3 - Training of the Siamese ConvNet with Weight-Sharing from part 2 with different auxiliary coefficients.")
print("         The bigger the coefficient, the more the digits' predictions are taken into account:\n")

print("For the test.py file, we will only run our best model (with an auxiliary coefficient of 100)the others values are loaded")
print("Note that the new results might not be the best results, but should be close...")

print("Loading the results...")
AuxiliaryConvNet = torch.load('results/AuxiliaryConvNet.pt')

best_coef = 100
convnet_config = [ [1, 128, 32], 50, True] # [ channels, hidden, digit ]
config = [ConvNet, convnet_config, weight_sharing, best_coef] # [Model_fn, config, weight_sharing, aux_coeff]
model = AuxModel(*config)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
num_params = compute_number_parameters(model)
print("Training a ConvNet in a Siamese fashion with {} parameters, with Weight-Sharing and an auxiliary coefficient of {}...".format(num_params, best_coef))
mean, std, metrics = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
print("Training over...\n")

# Replace the already computed results by the results we just computed
AuxiliaryConvNet['AuxConvNet_'+str(best_coef)]['mean'] = mean
AuxiliaryConvNet['AuxConvNet_'+str(best_coef)]['std'] = std
AuxiliaryConvNet['AuxConvNet_'+str(best_coef)]['metrics'] = metrics
AuxiliaryConvNet['AuxConvNet_'+str(best_coef)]['num_params'] = num_params

# print the results
aux_coeffs = [1e-2, 1e-1, 1, 2, 5, 10, 100]

for aux_coef in aux_coeffs:
    
    mean = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['mean']
    std = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['std']
    num_params = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['num_params']
    print(f"For a ConvNet with {num_params} parameters, Weight-Sharing and {aux_coef} auxiliary coefficient:")
    if aux_coef == best_coef:
        print("This is the model we just trained")
    print("\tMean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))

