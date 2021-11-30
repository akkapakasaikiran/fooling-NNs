import numpy as np
import torch
from torch.nn.functional import softmax
import os

from utils import set_seeds
from attack import attack
from utils import show_image, to4d

def expt1(model, test_data, device, num_classes=10, figs_dir=None):
	""" See relation between ground truth and false prediction class. """

	classes_seen = set()
	norm_rs = np.zeros((num_classes, num_classes))
	probs = np.zeros((num_classes, num_classes))
	for i in range(100):
		if len(classes_seen) == num_classes: break
		
		X, y = test_data[i]
		if y in classes_seen: continue
		else: classes_seen.add(y)

		for false_pred in range(num_classes):
			fig_file = None
			if figs_dir:
				fig_name = f'{model.get_name()}_{y}_{false_pred}'
				fig_file = os.path.join(figs_dir, fig_name)

			norm_r, prob = attack(model, X, y, false_pred, device, fig_file)

			norm_rs[y][false_pred] = norm_r
			probs[y][false_pred] = prob

	return norm_rs, probs

def show_expt1_orig_images(test_data, num_classes=10, figs_dir=None):
	classes_seen = set()
	for i in range(100):
		if len(classes_seen) == num_classes: break
		
		X, y = test_data[i]
		if y in classes_seen: continue
		else: classes_seen.add(y)

		fig_file = os.path.join(figs_dir, f'{y}')
		show_image(X, fig_file)



def expt2(model, test_data, device, figs_dir=None):
	""" Compare the use of L1 and L2 penalty on the perturbation r. """
	print('L1 norm')
	for num in range(3):
		X, y = test_data[num]
		false = 10 - y
		fig_file = os.path.join(figs_dir, f'l1-{num}')
		print(attack(model, X, y, false, device, fig_file=fig_file, norm='l1'))


	print('L2 norm')
	for num in range(3):
		X, y = test_data[num]
		false = 10 - y
		fig_file = os.path.join(figs_dir, f'l2-{num}')
		print(attack(model, X, y, false, device, fig_file=fig_file, norm='l2'))


def expt3(models, X, y, device):
	f = 10 - y
	n = len(models)
	results = np.zeros((n,n,7))
	for i, m1 in enumerate(models):
		print(f'{i = }, m1 = {m1.get_name()}')
		r, p = attack(m1, X, y, f, device)
		for j, m2 in enumerate(models):
			rs = [0.8*r, r, 1.2*r, 1.5*r, 2*r, 4*r]
			ps = []
			for rr in rs:
				Xr = torch.clamp(X + r, 0, 1).to(device)
				if len(X.shape) == 3: X = to4d(X)
				logits = m2(Xr)
				p = softmax(logits, dim=1).detach().cpu().numpy()[0][f]
				ps.append(p)
			results[i][j] = np.array(p)

	return results
			
		
def expt4(model, X, y, device):
	f = 10 - y
	r, p = attack(model, X, y, f, device)
	rs = [0.5*r, 0.8*r, r, 1.2*r, 1.5*r, 2*r]
	ps = []
	for r in rs:
		Xr = torch.clamp(X + r, 0, 1).to(device)
		if len(Xr.shape) == 3: X = to4d(X)
		logits = model(Xr)
		p = softmax(logits, dim=1).detach().cpu().numpy()[0][f]
		ps.append(p)
	return ps
