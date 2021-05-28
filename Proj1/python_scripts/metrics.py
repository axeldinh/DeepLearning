class Metrics():
    """
    Little class containing the desired metrics, for now it can take:
        -train_losses
        -test_losses
        -accuracy
    Each of them consist in a list of lists containing the metrics at each epoch and training, e.g
    [ [loss_epoch1_training1, ..., loss_epoch25_training1], ..., [loss_epoch1_training10, ..., loss_epoch25_training10] ]
    """
    
    def __init__(self):
        
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []