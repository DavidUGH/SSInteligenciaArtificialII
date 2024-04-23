import numpy
import torch
import torch.nn as nn
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeavePOut, LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class MLP(nn.Module):
    def __init__(self):
        input_size = 4
        hidden_size = 30
        output_size = 3
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        #print(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.hardtanh(out)
        #print(out)
        return out

file = pandas.read_csv('irisbin.csv', header=None)
X = file.iloc[:, :-3].values
y = file.iloc[:, -3:].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

xtraintensor = torch.tensor(xtrain, dtype=torch.float32)
ytraintensor = torch.tensor(ytrain, dtype=torch.float32)
xtestsensor = torch.tensor(xtest, dtype=torch.float32)
ytesttensor = torch.tensor(ytest, dtype=torch.float32)


m = MLP()

c = nn.CrossEntropyLoss()
LR = 0.01
op = torch.optim.Adam(m.parameters(), LR)

total_error = 0.0
correct = 0

epochs=2
loo = LeavePOut(1)

errors1 = []
for train_index, test_index in loo.split(xtrain):
    xtrainfold, xvalfold = xtrain[train_index], xtrain[test_index]
    ytrainfold, yvalfold = ytrain[train_index], ytrain[test_index]

    xtrainfold_tensor = torch.tensor(xtrainfold, dtype=torch.float32)
    ytrainfold_tensor = torch.tensor(ytrainfold, dtype=torch.int32)
    xvalfold_tensor = torch.tensor(xvalfold, dtype=torch.float32)
    yvalfold_tensor = torch.tensor(yvalfold, dtype=torch.int32)

    for epoch in range(epochs):
        op.zero_grad()
        outputs = m(xtrainfold_tensor)
        loss = c(outputs, ytrainfold_tensor.argmax(dim=1).long())
        loss.backward()
        op.step()
        errors1.append(loss.item())
        print(f'Fold [{test_index[0]+1}], Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

predicted_classes = []

with torch.no_grad():
    outputs = m(xtestsensor)
    print(outputs)
    yvalues, predicted = torch.max(outputs, 1)
    print(yvalues)
    predicted_classes.append(predicted)
    correct = (predicted == torch.max(ytesttensor, 1)[1]).sum().item()
    total = ytesttensor.size(0)
    accuracy = correct / total
    o = ''
    for a in range(len(xtestsensor)):
        o += 'IN: '
        o += str(xtestsensor[a])
        o += ' EXPECTED: '
        o += str(ytesttensor[a])
        o += ' REAL: '
        o += str(outputs[a])
        o += '\n'
    print(o)
    print(f'Accuracy: {accuracy:.2f}')

m2 = MLP()

c = nn.CrossEntropyLoss()
LR = 0.01
op = torch.optim.Adam(m.parameters(), LR)

total_error = 0.0
correct = 0

epochs=1
k = 2
loo = LeavePOut(k)

errors2 = []
for train_index, test_index in loo.split(xtrain):
    xtrainfold, xvalfold = xtrain[train_index], xtrain[test_index]
    ytrainfold, yvalfold = ytrain[train_index], ytrain[test_index]

    xtrainfold_tensor = torch.tensor(xtrainfold, dtype=torch.float32)
    ytrainfold_tensor = torch.tensor(ytrainfold, dtype=torch.int32)
    xvalfold_tensor = torch.tensor(xvalfold, dtype=torch.float32)
    yvalfold_tensor = torch.tensor(yvalfold, dtype=torch.int32)

    for epoch in range(epochs):
        op.zero_grad()
        outputs = m2(xtrainfold_tensor)
        loss2 = c(outputs, ytrainfold_tensor.argmax(dim=1).long())
        loss2.backward()
        op.step()
        errors2.append(loss.item())
        print(f'Fold [{test_index[0]+1}], Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

predicted_classes = []

with torch.no_grad():
    outputs = m2(xtestsensor)
    print(outputs)
    yvalues, predicted = torch.max(outputs, 1)
    print(yvalues)
    predicted_classes.append(predicted)
    correct = (predicted == torch.max(ytesttensor, 1)[1]).sum().item()
    total = ytesttensor.size(0)
    accuracy = correct / total
    o = ''
    for a in range(len(xtestsensor)):
        o += 'IN: '
        o += str(xtestsensor[a])
        o += ' EXPECTED: '
        o += str(ytesttensor[a])
        o += ' REAL: '
        o += str(outputs[a])
        o += '\n'
    print(o)
    print(f'Accuracy: {accuracy:.2f}')


plt.figure() # Error chart
plt.scatter(range(len(errors1)) , errors1)
title = "Leave k out error char\nStandard Deviation:t" + str(numpy.std([errors1]))
plt.title(title)

plt.figure() # Actual chart
plt.scatter(range(len(errors2)) , errors2)
title = "Leave one out error chart\nStandard Deviation:" + str(numpy.std([errors2]))
plt.title(title)

plt.show()
