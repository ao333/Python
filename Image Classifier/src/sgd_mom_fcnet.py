import numpy as np
from scipy.misc import imread
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import matplotlib.pyplot as plt
from os import makedirs,path
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
hyper_parameter_list = {
	'regularization': np.array([base*(10**expo) for base in (1,2,4,8) for expo in (-1, -2, -3,-4)]+[0]),
	'hidden_layer': sorted([[50*layer1*layer2, 50*layer1] for layer1 in range(1,13) for layer2 in [1,2]], key=lambda item:item[0]),
	'momentum' :[0.05*entry for entry in range(21)],
	'learning_rate': np.array([base*(10**expo) for base in (2,4,8,10) for expo in (-3,-4, -5)]),
	'lr_decay': [0.5 + (entry*0.05) for entry in range(11)],
	'num_epochs':[10, 15, 20, 25,30],
	'batch_size': [32, 64, 128, 256],
	'num_training': [2000*entry for entry in range(1,12)],
	'dropout': [0.1*entry for entry in range(0,10)]
}


data = dict()
#Iterate all the hyper-paramter type in hyper_parameter_list
for para_type, value_list in hyper_parameter_list.items():
	default_para = {
		'regularization': 0,
		'hidden_layer': [300,300],
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


	print('***************************************************************')
	print('Parameter type to test is: ',para_type)
	print('***************************************************************')

	#Create Folder to store the plotted graph for corresponding parameter type
	try:
		makedirs(para_type)
	except OSError as exc:
		pass


	plot_para_value = []#Store the x-axis value (parameter value)
	plot_val_acc = []#Store the y-axis value (validation accuracy)


	for value in value_list:#Interate all the value for specific hyper-parameter
		default_para[para_type] = value
		if para_type == 'learning_rate':
			default_para['regularization'] = 0

		#Hyper-parameter Display
		print('*******************')
		print(para_type, ':', value)
		print('*****************************************')
		print('The hyper-parameter for training is') 
		for k, v in default_para.items():
			print(k, ':', v)
		print('*****************************************')


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

		#For 'learning_rate' and 'regularization', the x-axis is in log-space
		if (para_type == 'learning_rate' or para_type =='regularization') and value != 0:
			plot_para_value.append(np.log10(value))
		else:
			plot_para_value.append(value)

		plot_val_acc.append(best_val_acc)

		#Calculate the Confusion Matrix, F1 and print out them
		confu_mat, F1 = solver.get_conf_mat_F1(type='validation')
		print('\n')
		print('**************The best validation data Accuracy is: ', best_val_acc, '***********************')
		print('F1 value: ', F1)
		print('Confusion Matrix: \n')
		display_conf(confu_mat)
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
		plt.savefig(para_type+'/'+str(value)+'.png')
		plt.close()

	#Plot Graph: Hyper-parameter value Vs. Validation Accuracy
	plt.subplot(1,1,1)
	if (para_type == 'hidden_layer'):
		plot_para_value = list(range(1,len(value_list)+1))
		plt.xticks(plot_para_value, [str(entry) for entry in value_list])
		plt.xticks(rotation=90)
	plt.title(para_type+' vs Accuracy')
	plt.plot(plot_para_value, plot_val_acc,'o',label='Validation Best Accuracy')
	for i,j in zip(plot_para_value, plot_val_acc):
			plt.annotate(str(j),xy=(i,j))
	if para_type == 'learning_rate' or para_type =='regularization':
			plt.xlabel('log '+para_type)
	else: 
		plt.xlabel(para_type)
	plt.legend(loc='lower right')
	plt.gcf().set_size_inches(15,12)
	plt.savefig(para_type+'/'+'overall.png')
	plt.close()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

