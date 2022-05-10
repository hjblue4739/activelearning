import numpy as np
from numpy.random import choice, normal
np.random.seed(8)
from tensorflow import set_random_seed
set_random_seed(8)

from tensorflow.python.keras.layers import Input, BatchNormalization,Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Reshape
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping

from keras.datasets import mnist

#load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scale and reshpae training data
x_train = x_train.reshape(-1,28,28,1) / 255

n_samples = len(x_train)

#hyperparameters for auto labeling

#init_label is the initial percent of data that is hand labeled
init_label = 0.01

#high_confidence_threshold is the minimum classification probability that..
#..classifier must predict for the sample to be auto labeled 
high_confidence_threshold = 0.9999

#low_confidence_percent is the percentage of unlabeled data with lowest..
#..classification probabilities that are hand labeled each iteration 
low_confidence_percent = 0.01

#select data as initial hand labeled samples
x_labeled = x_train[:round(len(x_train)*init_label)]
y_labeled = y_train[:round(len(x_train)*init_label)]

#store the rest of the data as unlabeled data
x_unlabeled = x_train[round(len(x_train)*init_label):]
y_unlabeled = y_train[round(len(x_train)*init_label):]

#empty arrays for storing auto labeles and correct labeld for comparing accuracy
auto_labels = np.array([])
correct_labels = np.array([])

n_hand_labeled = len(x_labeled)

#set up new classifier
def compile_model():
    inputs = Input(shape=(28,28,1))
    net = inputs
    net = Conv2D(32, 3, activation='relu')(net)
    net = Conv2D(64, 3, activation='relu')(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Dropout(0.25)(net)

    net = Flatten()(net)

    net = Dense(128, activation='relu')(net)
    outputs = Dense(10, 'softmax')(net)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='Adadelta',loss='sparse_categorical_crossentropy',
		  metrics=['accuracy'])
    
    return model
    
iteration=0
run = True

#algorithm loops until less than 1% of samples are unlabeled
while run == True:
    print(iteration)
    model = compile_model()
    
    #early stopping when validation accuracy stops increasing for 10 epochs
    earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10,
		verbose=0, mode='max', baseline=None, restore_best_weights=True)
    
    #fit model on labeled data using 20% of labeled data as validation
    model.fit(x=x_labeled, y=y_labeled, batch_size=32, epochs=99999, verbose=0,
	      	callbacks=[earlyStop], validation_split=0.2)
    
    #use model to predict labels for all unlabeled data
    predictions = model.predict(x_unlabeled)
    
    #prediction indices sorted by classsification probabilty
    sorted_indices = np.argsort(predictions.max(axis=1))
    
    #indices of unlabeled data that are going to be hand labeled..
    #..and added to labeled data set
    low_confidence = sorted_indices[:round(n_samples*low_confidence_percent)]
    x_labeled = np.concatenate([x_labeled,x_unlabeled[low_confidence]])
    y_labeled = np.concatenate([y_labeled,y_unlabeled[low_confidence]])
    
    #indices of unlabeled data that are going to be..
    #..auto labeled and added to data set
    high_confidence = np.array([x for x
	in np.argwhere(predictions.max(axis=1)>high_confidence_threshold).flatten()
	if x not in low_confidence])
    if len(high_confidence) > 0:
        x_labeled = np.concatenate([x_labeled,
		x_unlabeled[high_confidence]])
        y_labeled = np.concatenate([y_labeled,
		predictions[high_confidence].argmax(axis=1)])
    
        correct_labels = np.concatenate([correct_labels,
		y_unlabeled[high_confidence]])
        auto_labels = np.concatenate([auto_labels,
		predictions[high_confidence].argmax(axis=1)])
        
        #remove labeled dat from unlabeled data set
        x_unlabeled = np.delete(x_unlabeled,
		np.concatenate([low_confidence,high_confidence]),0)
        y_unlabeled = np.delete(y_unlabeled,
		np.concatenate([low_confidence,high_confidence]),0)
    
    else:
    	#remove labeled dat from unlabeled data set
        x_unlabeled = np.delete(x_unlabeled,low_confidence,0)
        y_unlabeled = np.delete(y_unlabeled,low_confidence,0)
    
    total_auto_labeled = len(correct_labels)
    total_incorrect_labels = total_auto_labeled - 
	np.count_nonzero(correct_labels == auto_labels)

    n_hand_labeled += len(low_confidence)
    
    print('after iteration '+str(iteration)+':',len(x_labeled),
	  'labeled samples,',len(x_unlabeled),'unlabeled samples')
    print(str(len(low_confidence))+' hand labeled')
    print(str(len(high_confidence))+' auto labeled')
	print(str(len(y_unlabeled)/n_samples)+'% unlabeled')
    print(str(total_incorrect_labels/total_auto_labeled)+
	  		'% incorrectly labeled')
    
    if len(y_unlabeled)/n_samples < 0.01:
        run=False
    
    iteration+=1
    
print('total hand labeled:',n_hand_labeled)
print('total auto labeled:',total_auto_labeled+
        len(y_unlabeled))
print(str(total_incorrect_labels/(total_auto_labeled+
	len(y_unlabeled)))+'% incorrectly labeled')


'''
Here are the results from several trials with different hyper parameters.
Initial hand label: The proportion of data initially hand labeled in step 1
Upper probability threshold: The minimum classification probability that the classifier must predict for a sample to be auto labeled
Lower probability proportion: The percentage of unlabeled data with lowest classification probabilities that are hand labeled in step 5
Number of iterations: The number of times steps 2–5 are repeated
We can see that there is a trade off between number of hand labeled data samples and proportion of mislabeled samples. An upper probability threshold of 0.999999 produced the least number of mislabeled samples but took the most iterations to complete.

Future Work
There are many altercations we could make to the presented active learning algorithm for label sampling. For example, instead of auto-labeling samples with a class probability above a certain threshold, we can set a fixed number of samples to auto-label during step 4. Also, we could auto label the remaining labels in step 7 instead of hand labeling them. Experimenting with other variations may results in a more ultimate algorithm. Additionally, we can explore the results of active learning on more complex data sets or data sets with unbalanced classes.
'''
