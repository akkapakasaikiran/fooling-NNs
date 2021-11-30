import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def set_seeds(seed=0):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


# Useful when you want to feed a single image into a CNN
def to4d(data):
	""" shape [a,b,c] -> shape [1,a,b,c]. """
	return data.unsqueeze(0)

def name_of_class(pred):
	""" Return name of FashionMNIST class. """
	classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
	return classes[pred]

def show_image(data, fig_file='figs/fig.png'):
	""" Plot data as an image. """
	data = data.numpy()
	fig = plt.figure(figsize=(2,2))
	plt.imshow(data.reshape(28, 28), cmap='gray')
	plt.axis('off')
	plt.savefig(fig_file)
	plt.show()
	plt.close()
	