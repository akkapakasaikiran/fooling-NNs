from torch import nn


class NeuralNetwork(nn.Module):
	def __init__(self, h1, h2, num_classes=10, name='NN'):
		super().__init__()
		self.name = name
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, h1), 
			nn.ReLU(),
			nn.Linear(h1, h2),
			nn.ReLU(), 
			nn.Linear(h2, 10)
		)

	def forward(self, x):
		x = self.flatten(x)
		x = self.linear_relu_stack(x)
		return x

	def get_name(self): return self.name

	def get_type(self): return 'NeuralNetwork'

	def num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	
class CNN(nn.Module):
	def __init__(self, h1=64, h2=128, input_size=28, num_classes=10, name='CNN'):
		super().__init__()
		self.name = name
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, h1, 5, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(h1, h2, 5, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)    
		) 
		self.flatten = nn.Flatten()
		num_neurons = h2 * (input_size // (2*2))**2
		self.fc = nn.Linear(num_neurons, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.flatten(x)
		x = self.fc(x)
		return x

	def get_name(self): return self.name

	def get_type(self): return 'CNN'

	def num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
