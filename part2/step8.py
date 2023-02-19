import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

#Step 2
#Data parsing
import lib


#Step 8
#PyTorch
f = 40
#distance between points: pi/5
t = np.linspace(0,2*np.pi,10)
sinseq = []
cosseq = []
# fig, axs = plt.subplots(10,2)
# fig.set_figheight(45)
# fig.set_figwidth(15)
for i in range(1000):
    A = np.random.rand()
    p = np.random.rand()*2*np.pi
    sinseq.append(A*np.sin(2*np.pi*f*t + p))
#     axs[i][0].plot(t, sinseq[i])
#     axs[i][0].set_title('Sine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2))+ ' rad')
    cosseq.append(A*np.cos(2*np.pi*f*t + p))
#     axs[i][1].plot(t, cosseq[i])
#     axs[i][1].set_title('Cosine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2)) + ' rad')
fig, axs = plt.subplots(1,2)
fig.set_figheight(2)
fig.set_figwidth(10)
axs[0].plot(t, sinseq[999])
axs[0].set_title('Sine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2))+ ' rad')
axs[1].plot(t, cosseq[999])
axs[1].set_title('Cosine with amplitude: '+str(np.round(A,2))+'  and  phase: ' + str(np.round(p,2)) + ' rad')
plt.savefig('../plots/waveforms.png')
# plt.show()

X = np.array(sinseq)
y = np.array(cosseq)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)



batch_size = X_train.shape[0] # how many mini-batches, each mini is a sine
seq_length = X_train.shape[1] # length of each mini-batch
input_sz = 1 # feed 1 point at a time
hidden_dim = 1

#Initialize our model
# rnn = torch.nn.RNN(input_size = input_sz, hidden_size = hidden_dim, num_layers = 1, batch_first = True).double()
rnn = torch.nn.LSTM(input_size = input_sz, hidden_size = hidden_dim, num_layers = 1, batch_first = True).double()
h_t = torch.zeros(1, X_train.shape[0], hidden_dim, dtype=torch.double)
c_t = torch.zeros(1, X_train.shape[0], hidden_dim, dtype=torch.double)

#input shape : (batch_size, seq_length, input_size)
X_train = X_train.view(batch_size, seq_length, input_sz)
X_test = X_test.view(X_test.shape[0], seq_length, input_sz)
y_train = y_train.view(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.view(y_test.shape[0], y_test.shape[1], 1)

#out shape : (batch_size, seq_length, hidden_dim)
#h_n shape : (num_layers, batch_size, hidden_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
epochs = 4500

for i in range(epochs):
	optimizer.zero_grad()
	
# 	out, h_n = rnn(X_train) #forward pass for RNN
	out, h_n = rnn(X_train, (h_t,c_t)) #forward pass for LSTM

	#############	#############	#############	#############	#############
	#If hidden_dim > 1 :
	#This is what they say on a github issue, but performs worse!!
# 	output_layer = torch.nn.Linear(hidden_dim, 1).double() #convert hidden_dim size
# 	out = output_layer(out)								   #to target output size : 1
	#############	#############	#############	#############	#############

	loss = criterion(out, y_train) #calculate loss
	loss.backward() #back propagation
	if i%300==0:
		print('Training loss:', loss.item())
	optimizer.step()

	with torch.no_grad():
		pred, h_next = rnn(X_test)
		loss = criterion(pred, y_test)
		if i%300==0:
			print('Test loss:', loss.item())

with torch.no_grad():
    pred, h_next = rnn(X_test.view(X_test.shape[0], seq_length, input_sz))

    	#############	#############	#############	#############
# output_layer = torch.nn.Linear(hidden_dim, 1).double() #convert hidden_dim size
# pred = output_layer(pred)					   		   #to target output size : 1
		#############	#############	#############	#############

fig = plt.figure(figsize=(25,10))
for i in range(15):
    fig.add_subplot(3, 5, i+1)
    plt.axis('off')
    output = pred[i*5].view(10).detach().numpy()
    plt.plot(np.arange(10), output, 'g')
    plt.plot(np.arange(10), y_test[i*5], 'r')
plt.savefig('../plots/predictions8.png')
# plt.show()
