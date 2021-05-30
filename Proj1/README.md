
# Project 1 

For this project, the objective is to compare different architectures of neural networks whose purpose are to recognize if one digit image is higher than another digit image.  Two main categories of architectures have been studied here.  Networks that treat a pair of images as one image with two channels (one channel corresponds to one image) or networks that analysis each image separately before combining the information into one output (siamese network).  Additional features such as weight sharing and auxiliary losses were also studied to assess if any improvement can be achieved when adding them.

To run the classical models implemented here with their top hyperparameters run the test.py file (Under 10 min run on the virtual machine).

Following libraries are required:
- Pytorch

To generate the graphs, run the gen_figures_results.py (about 1 hour run). If you do not want to wait that long, please set Load_1/Load_2/Load_3 to True, this will load already computed results and show them/make the plots.

Additional libraries would be needed:
- Matplotlib
