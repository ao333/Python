import numpy as np
from scipy.misc import imread
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import matplotlib.pyplot as plt
from os import makedirs,path
import pickle
"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
#Formatted Confusion maxtrix display
def display_conf(confu):
	label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	n_label = len(confu)
	print(' ',end=",")
	for index in range(n_label):
		if index != n_label-1:
			print(label[index], end=',')
		else:
			print(label[index])
	for index1 in range(n_label):
		print(label[index1], end=',')
		for index2 in range(n_label):
			if index2 != n_label-1:
				print(int(confu[index1][index2]), end=",")
			else:
				print(int(confu[index1][index2]))
#Hyper Parameter value to be tested


data = dict()
#Iterate all the hyper-paramter type in hyper_parameter_list

default_para = {
'regularization': 0,
'hidden_layer': [450,450],
'momentum': 0.95,
'learning_rate': 4e-4,
'update_rule': 'sgd_momentum',
'lr_decay': 0.9,
'num_epochs': 20,
'batch_size': 64,
'num_training': 22000,
'num_validation': 2000,
'num_test': 2000,
'dropout': 0.3
}



data = get_FER2013_data(num_training=default_para['num_training'], 
	num_validation=default_para['num_validation'], 
	num_test=default_para['num_test'], subtract_mean=True)
model = FullyConnectedNet(default_para['hidden_layer'], 
							input_dim=48*48, num_classes=7, 
							reg = default_para['regularization'],
							dropout = default_para['dropout'])
solver = Solver(model, data,
				update_rule=default_para['update_rule'],
				optim_config={
				  'learning_rate': default_para['learning_rate'],
				  'momentum': default_para['momentum']
				},
				lr_decay=default_para['lr_decay'],
				num_epochs=default_para['num_epochs'], 
				batch_size=default_para['batch_size'] ,
				print_every=200,
				verbose = True)
solver.train()

best_val_acc = solver.best_val_acc


#Calculate the Confusion Matrix, F1 and print out them
valid_confu_mat, valid_F1 = solver.get_conf_mat_F1(type='validation')
test_confu_mat, test_F1 = solver.get_conf_mat_F1(type='test')
print('\n')
print('**************The best validation data Accuracy is: ', best_val_acc, '***********************')
print('**************The test data Accuracy is: ', solver.get_test_accu(), '***********************')
print('F1 value for Validation: ', valid_F1)
print('Confusion Matrix for Validation: \n')
display_conf(valid_confu_mat)
print('F1 value for Test: ', test_F1)
print('Confusion Matrix for Test: \n')
display_conf(test_confu_mat)
print('\n\n\n')

#Plot graph: Loss VS. Iteration, Accuracy VS. Epoch
plt.subplot(2,1,1)
plt.title('Trainingloss')
plt.plot(solver.loss_history,'o')
plt.xlabel('Iteration')
plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history,'-o',label='train')
plt.plot(solver.val_acc_history,'-o',label='val')
plt.plot([0.5]*len(solver.val_acc_history),'k--')
for i,j in zip(np.arange(len(solver.train_acc_history)),solver.train_acc_history):
	plt.annotate(str(j),xy=(i,j))
for i,j in zip(np.arange(len(solver.val_acc_history)),solver.val_acc_history):
	plt.annotate(str(j),xy=(i,j))
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
plt.savefig('best_para.png')
plt.close()


pickle.dump(solver.model, open( "best_model.p", "wb" ) )


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

