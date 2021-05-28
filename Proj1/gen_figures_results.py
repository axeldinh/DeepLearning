from python_scripts import *
import matplotlib.pyplot as plt

# Same script as test.py but with results and figures generation
# might take more time to run, as we make more experiences

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Booleans to load or not the different parts
load_1 = True
load_2 = True
load_3 = True

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

# Training and/or loading of the MLP
if not load_1:
    
    Full_MLP_ConvNet = {'MLP': {}, 'ConvNet': {}} # Dictionnary to store results
    
    config = [ [392, 200, 100, 2], [0.0, 0.0], False] # [ layers_sizes, dropouts, digit ]
    model = MLP(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a MLP with {} parameters...".format(num_params))
    mean, std, metrics = get_mean_std(MLP, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
    print("Training over...\n")
    
    # Store the results
    Full_MLP_ConvNet['MLP']['mean'] = mean
    Full_MLP_ConvNet['MLP']['std'] = std
    Full_MLP_ConvNet['MLP']['metrics'] = metrics
    Full_MLP_ConvNet['MLP']['num_params'] = num_params


# Training and/or loading of the ConvNet
if not load_1:
    config = [ [2, 128, 32], 400, False] # [ channels, hidden, digit ]
    model = ConvNet(*config)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_params = compute_number_parameters(model)
    print("Training a ConvNet with {} parameters...".format(num_params))
    mean, std, metrics = get_mean_std(ConvNet, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
    print("Training over...\n")
    
    # Store the results
    Full_MLP_ConvNet['ConvNet']['mean'] = mean
    Full_MLP_ConvNet['ConvNet']['std'] = std
    Full_MLP_ConvNet['ConvNet']['metrics'] = metrics
    Full_MLP_ConvNet['ConvNet']['num_params'] = num_params

    # Save the results
    torch.save(Full_MLP_ConvNet, 'results/Full_MLP_ConvNet.pt')

print("Loading results...\n")
Full_MLP_ConvNet = torch.load('results/Full_MLP_ConvNet.pt')

for model in ['MLP', 'ConvNet']:
    mean = Full_MLP_ConvNet[model]['mean']
    std = Full_MLP_ConvNet[model]['std']
    num_params = Full_MLP_ConvNet[model]['num_params']
    
    print(f"For a {model} with {num_params} parameters:")
    print("Mean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))

#######################
# 2-Run a MLP and a ConvNet on 1 digit, in a Siamese fashion, with a small MLP for the decision making
#   We either declare 2 MLP/ConvNet for each digit or 1 for both to see the effect of weight-sharing

print("#######################################################")
print("Part 2 - Training of a MLP and a ConvNet trained on the AuxModel (the Siamese model) with/without weight-sharing, but no auxiliary losses:\n")

# Training of a MLP in AuxModel:

if not load_2:
    
    SiameseWeight_sharing = {'MLP_False': {}, 'MLP_True': {}, 'ConvNet_False': {}, 'ConvNet_True': {}} # Dictionary to store the results
    
    # Training the MLPs
    for weight_sharing in [False, True]:
        
        mlp_config = [ [196, 150, 100, 10], [0.0, 0.0], True] # [layers_sizes, dropouts, digit]
        config = [MLP, mlp_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
        model = AuxModel(*config)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        num_params = compute_number_parameters(model)
        print("Training a MLP in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
        mean, std, metrics = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
        print("Training over...\n")
        
        SiameseWeight_sharing["MLP_"+str(weight_sharing)]['mean'] = mean
        SiameseWeight_sharing["MLP_"+str(weight_sharing)]['std'] = std
        SiameseWeight_sharing["MLP_"+str(weight_sharing)]['metrics'] = metrics
        SiameseWeight_sharing["MLP_"+str(weight_sharing)]['num_params'] = num_params
        
    # Training the ConvNets
    for weight_sharing in [False, True]:
        
        convnet_config = [ [1, 128, 32], 50, True] # [ channels, hidden, digit ]
        config = [ConvNet, convnet_config, weight_sharing, 0.0] # [Model_fn, config, weight_sharing, aux_coeff]
        model = AuxModel(*config)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        num_params = compute_number_parameters(model)
        print("Training a ConvNet in a Siamese fashion with {} parameters, weight_sharing = {}...".format(num_params, weight_sharing))
        mean, std, metrics = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
        print("Training over...\n")
        
        SiameseWeight_sharing["ConvNet_"+str(weight_sharing)]['mean'] = mean
        SiameseWeight_sharing["ConvNet_"+str(weight_sharing)]['std'] = std
        SiameseWeight_sharing["ConvNet_"+str(weight_sharing)]['metrics'] = metrics
        SiameseWeight_sharing["ConvNet_"+str(weight_sharing)]['num_params'] = num_params
    
    torch.save(SiameseWeight_sharing, 'results/SiameseWeight_sharing.pt')

print("Loading results...\n")
SiameseWeight_sharing = torch.load('results/SiameseWeight_sharing.pt')

for model in ['MLP', 'ConvNet']:
    for weight_sharing in [False, True]:
        mean = SiameseWeight_sharing[model + '_' + str(weight_sharing)]['mean']
        std = SiameseWeight_sharing[model + '_' + str(weight_sharing)]['std']
        num_params = SiameseWeight_sharing[model + '_' + str(weight_sharing)]['num_params']
        
        print("For a {} with Weight-Sharing = {} and {} parameters:".format(model, weight_sharing, num_params))
        print("\tMean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))
        
# Generating a Boxplot of the results

all_best_accs = []
for model in ['MLP_', 'ConvNet_']:
  for ws in ['False', 'True']:
    metrics = SiameseWeight_sharing[model + ws]['metrics']
    accuracies = torch.Tensor(metrics.accuracies)
    best_accs = accuracies.max(dim=1).values.tolist()
    all_best_accs.append(best_accs)

plt.figure()
plt.title('Accuracies of models with or without weight-sharing (WS)')
labels = ['MLP No WS', 'MLP WS', 'ConvNet No WS', 'ConvNet WS']
plt.boxplot(all_best_accs, labels = labels)
plt.ylabel('Accuracy')

plt.savefig('figures/SiameseWeight_sharing.png', format = 'png')

#######################
# 3-Run of the Siamese ConvNet from part 2 with different auxiliary coefficients
#   The bigger the coefficient, the more the digits' predictions are taken into account

print("#######################################################")
print("Part 3 - Training of the Siamese ConvNet with Weight-Sharing from part 2 with different auxiliary coefficients.")
print("         The bigger the coefficient, the more the digits' predictions are taken into account:\n")

epochs = 50
aux_coeffs = [1e-2, 1e-1, 1, 2, 5, 10, 100]

if not load_3:
    
    AuxiliaryConvNet = {"AuxConvNet_" + str(x): {} for x in aux_coeffs} # To store the results
    for aux_coef in aux_coeffs:
        
        convnet_config = [ [1, 128, 32], 50, True] # [ channels, hidden, digit ]
        config = [ConvNet, convnet_config, weight_sharing, aux_coef] # [Model_fn, config, weight_sharing, aux_coeff]
        model = AuxModel(*config)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
        num_params = compute_number_parameters(model)
        print("Training a ConvNet in a Siamese fashion with {} parameters, with Weight-Sharing and {} auxiliary coefficient...".format(num_params, aux_coef))
        mean, std, metrics = get_mean_std(AuxModel, config, optimizer, mini_batch_size, num_trains, epochs = epochs, device = device)
        print("Training over...\n")
        
        AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['mean'] = mean
        AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['std'] = std
        AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['metrics'] = metrics
        AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['num_params'] = num_params
        
    torch.save(AuxiliaryConvNet, 'results/AuxiliaryConvNet.pt')
    
print("Loading the results...\n")
AuxiliaryConvNet = torch.load('results/AuxiliaryConvNet.pt')

# Printing the results
for aux_coef in aux_coeffs:
    
    mean = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['mean']
    std = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['std']
    num_params = AuxiliaryConvNet['AuxConvNet_'+str(aux_coef)]['num_params']
    
    print(f"For a ConvNet with {num_params} parameters, Weight-Sharing and {aux_coef} auxiliary coefficient:")
    print("\tMean best test accuracy = {:.2f}%(+-{:.2f})\n".format(mean*100, std*100))

# Generating a lineplot with errorbars for the means of the accuracies obtained
means = []
stds = []

for coef in aux_coeffs:
    mean = AuxiliaryConvNet['AuxConvNet_' + str(coef)]['mean'].item()
    std = AuxiliaryConvNet['AuxConvNet_' + str(coef)]['std'].item()
    means.append(mean)
    stds.append(std)

plt.figure()
plt.title('Best Mean Accuracy VS Auxiliary Coefficient')
plt.errorbar(aux_coeffs, means, stds)
plt.xscale('log')
plt.xlabel('Auxiliary Coefficient')
plt.ylabel('Accuracy')

plt.savefig('figures/AuxiliaryConvNet.png', format = 'png')


#######################
# 4-Generating a lineplot of the accuracies across epochs of our best models
#   

# Retrieve the results
accs = []

files = ['Full_MLP_ConvNet.pt', 'Full_MLP_ConvNet.pt', 'SiameseWeight_sharing.pt', 'AuxiliaryConvNet.pt']
models = ['MLP', 'ConvNet', 'ConvNet_True', None]

for file_name, model in zip(files, models):
      results = torch.load('results/' + file_name)
      if file_name == 'AuxiliaryConvNet.pt':
      # Get the best model with auxiliary losses
          best_mean_acc = 0
          for key in results.keys():
              if results[key]['mean'] > best_mean_acc:
                  best_mean_acc = results[key]['mean']
                  acc = torch.Tensor(results[key]['metrics'].accuracies).mean(dim = 0)
                  aux_coef = key.split('_')[1]
      else:
          # We average the accuracies accros the epochs (we get average accuracies per epoch)
          acc = torch.Tensor(results[model]['metrics'].accuracies).mean(dim = 0)
      
      accs.append(acc)

# Plot the results
plt.figure()
for acc in accs:
    plt.plot(range(1, len(acc)+1), acc)

plt.title('Accuracy during Epochs')
plt.legend(['Naive MLP', 'Naive ConvNet', 'Siamese ConvNet with WS',f"Siamese ConvNet with WS\n and Auxiliary Coefficient = {aux_coef}"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig('figures/AccuraciesBestModels.png', format = 'png')