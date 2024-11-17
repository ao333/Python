import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver1 import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = dict()
data = get_CIFAR10_data(50,50)
model = FullyConnectedNet([2000,2000], reg = 1e-3)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
solver.train()

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.title('Trainingloss')
plt.plot(solver.loss_history,'o')
plt.xlabel('Iteration')
plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history,'-o',label='train')
plt.plot(solver.val_acc_history,'-o',label='val')
plt.plot([0.5]*len(solver.val_acc_history),'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
plt.show()


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################