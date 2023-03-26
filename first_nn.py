'''Building a simple neural network'''

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas
from torch.autograd import Variable

from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, data):
        self.x = data.iloc[:,0:-1].values
        self.y = data.iloc[:,-1].values
        self.n_samples = data.shape[0]
        
    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float), torch.tensor(self.y[index], dtype=torch.long)
    
    def __len__(self):
        return len(self.x)
class IrisNet(nn.Module):  # neural network model

    def __init__(self,input_size,hidden1_size,hidden2_size,_num_classes):
        super(IrisNet,self).__init__()
        self.fc1=nn.Linear(input_size,hidden1_size)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(hidden1_size,hidden2_size)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(hidden2_size,_num_classes)

    def forward(self,x):
        out=self.fc1(x)
        out=self.relu1(out)
        out=self.fc2(out)
        out=self.relu2(out)
        out=self.fc3(out)
        return out
    
model=IrisNet(4,100,50,3)
print(model)
batch_size = 60


# Load the data
df = pandas.read_csv('iris.csv')

# # Reset the index

# Split the data into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Create the PyTorch data sets
train_ds = IrisDataset(train_df)
test_ds = IrisDataset(test_df)
# How many instances have we got?
print('# instances in training set: ', len(train_ds))
print('# instances in testing/validation set: ', len(test_ds))

# Create the dataloaders - for training and validation/testing
# We will be using the term validation and testing data interchangably
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)


''' A data loader is an essential component in PyTorch that allows you to load and iterate over your training, 
validation, or test datasets in batches. It takes a PyTorch dataset object as input and returns an iterator that can be used to iterate over the data in the dataset one batch at a time.'''

print(train_loader)


'''Instantiate the network, the loss function and the optimizer'''

# Our model
net = IrisNet(4, 100, 50, 3)

# Out loss function
criterion = nn.CrossEntropyLoss()

# Our optimizer (sto graident descent)
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)  

# Model Training
num_epochs = 500

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs): #total no. of epochs
    
    train_correct = 0
    train_total = 0

    for i, (items, classes) in enumerate(train_loader): # no. of iteration =total/batch
        
        # Convert torch tensor to Variable
        items = Variable(items)  # converting tensor to variable for calcualtion
        classes = Variable(classes)
        # print(items,classes)

        net.train()
        '''Put the network into training mode'''
        optimizer.zero_grad() # Clear off the gradients from any past operation
        outputs = net(items)  # Do the forward pass
        loss = criterion(outputs, classes) # Calculate the loss
        loss.backward()       # Calculate the gradients with help of back propagation
        optimizer.step()      # Ask the optimizer to adjust the parameters based on the gradients
        
        # Record the correct predictions for training data
        train_total += classes.size(0)    
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == classes.data).sum()
        
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
       .format(epoch+1, num_epochs, i+1, len(train_ds)//batch_size, loss.item()))

        # print ('Epoch %d/%d, Iteration %d/%d, Loss: %.4f' 
        #        %(epoch+1, num_epochs, i+1, len(train_ds)//batch_size, loss.data[0]))

    net.eval()                 # Put the network into evaluation mode
    
    # Book keeping
    # Record the loss
    # train_loss.append(loss.data[0])
    train_loss.append(loss.item())
    # What was our train accuracy?
    train_accuracy.append((100 * train_correct / train_total))

    # How did we do on the test set (the unseen set)
    # Record the correct predictions for test data
    test_items = torch.FloatTensor(test_df.iloc[:, 0:-1].values)
    test_classes = torch.LongTensor(test_df.iloc[:,-1].values)

    outputs = net(Variable(test_items))
    loss = criterion(outputs, Variable(test_classes))
    # test_loss.append(loss.data[0])
    test_loss.append(loss.item())
    _, predicted = torch.max(outputs.data, 1)
    total = test_classes.size(0)
    correct = (predicted == test_classes).sum()
    test_accuracy.append((100 * correct / total))

'''Plot loss vs iterations'''
fig = plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test loss')
plt.title("Train and Test Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(train_accuracy, label='train accuracy')
plt.plot(test_accuracy, label='test accuracy')
plt.title("Train and Test Accuracy")
plt.legend()
plt.show()


