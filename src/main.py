import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchvision import datasets, utils, transforms
import matplotlib.pyplot as plt
import os
import time

from models import NeuralNetwork, CNN
from attack import attack
from args import get_args
from utils import set_seeds
from experiments import expt1, expt2, expt3, expt4


def train(dataloader, model, optimizer):
	""" Train the model for one epoch. """ 
	size = len(dataloader.dataset)
	loss_fn = nn.CrossEntropyLoss()
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		logits = model(X)
		loss = loss_fn(logits, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 400 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>6f}  [{current:>5d}/{size:>5d}]")
			

def test(dataloader, model):
	""" Test the model. """
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	loss_fn = nn.CrossEntropyLoss()
	model.eval()
	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			logits = model(X)
			test_loss += loss_fn(logits, y).item()
			correct += (logits.argmax(1) == y).type(torch.float).sum().item()
	test_loss /= num_batches
	correct /= size
	print(f"Test: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>6f}")


if __name__ == '__main__':
	args = get_args()
	print(args)

	seeds = [0, 9, 42]
	set_seeds()

	train_data = datasets.FashionMNIST(
		root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.FashionMNIST(
		root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())

	train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
	test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

	device = torch.device(
		f'cuda:{args.cuda}' 
		if torch.cuda.is_available() and args.cuda != -1 else 'cpu'
	)
	print(f'{device = }')

	nn_small = NeuralNetwork(64, 128, name='NN-small').to(device)
	nn_large = NeuralNetwork(128, 256, name='NN-large').to(device)
	cnn_small = CNN(64, 128, name='CNN-small').to(device)
	cnn_large = CNN(128, 256, name='CNN-large').to(device)

	models = [nn_small, nn_large, cnn_small, cnn_large]

	# Restore model if saved previously, else train and store
	for model in models:
		model_name = f'{model.get_name()}.pth'
		model_file = os.path.join(args.checkpoint_dir, model_name)

		if os.path.isfile(model_file):
			model.load_state_dict(torch.load(model_file))
			print(f'Loaded model state from {model_file}')
			test(test_dataloader, model)

		else:
			optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
			
			start = time.perf_counter()
			print(f'Training model {model.get_name()}')
			epochs = 10
			for t in range(epochs):
				print(f"Epoch {t+1} ----------------")
				train(train_dataloader, model, optimizer)
				test(test_dataloader, model)
			end = time.perf_counter()
			print(f"Done! Time taken: {end - start}")

			torch.save(model.state_dict(), model_file)
			print(f'Stored model state to {model_file}')


	# seeds = [0, 9, 42]
	# norm_rs = np.zeros((10, 10))
	# probs = np.zeros((10, 10))

	# start = time.perf_counter()
	# for seed in seeds:
	# 	set_seeds(seed)
	# 	print(f'{seed = }')
	# 	tmp_a, tmp_b = expt1(nn_small, test_data, device)
	# 	norm_rs += tmp_a
	# 	probs += tmp_b
	# end = time.perf_counter()

	# norm_rs /= len(seeds)
	# probs /= len(seeds) 

	# print(f'Experiment 1 done! Time taken: {end - start}. Results:')
	# print(norm_rs)
	# print(probs)

	# expt2(nn_small, test_data, device, figs_dir=args.figs_dir)

	# n = len(models)
	# results = np.zeros((n, n, 7))
	# for num in range(3):
	# 	start = time.perf_counter()
	# 	X, y = test_data[num]
	# 	tmp = expt3(models, X, y, device)
	# 	results += tmp
	# 	end = time.perf_counter()
	# 	print(f'One run of expt3 done! Time taken: {end - start}')
	# results /= 3

	# print(results)

	# n = len(models)
	# ps = np.zeros((n, 6))
	# for num in range(3):
	# 	start = time.perf_counter()
	# 	X, y = test_data[num]
	# 		tmp = 
	# 		ps[i] += np.array(tmp)
	# 	end = time.perf_counter()
	# 	print(f'One run of expt4 done! Time taken: {end - start}')
	# ps /= 3

	# ps = expt4(model, X, y, device)

	# print(ps)
