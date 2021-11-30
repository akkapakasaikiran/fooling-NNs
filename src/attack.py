import torch
from torch import nn
from torch.nn.functional import softmax

from utils import to4d, show_image, set_seeds

def attack(
	model, X, y, false_pred, device, fig_file=None, 
	lamda=0.15, optim_steps=5000, norm='l2',
):
	print(f'Inside attack. {y = }, {false_pred = }.')
	X = X.to(device)
	if len(X.shape) == 3: X = to4d(X)

	r = torch.rand(1, 28, 28).to(device)
	r.requires_grad_()
	optimizer_r = torch.optim.Adam([r], lr=1e-3)
	loss_fn = nn.CrossEntropyLoss()

	for i in range(optim_steps):
		Xr = torch.clamp(X + r, 0, 1)  
		yf = torch.tensor([false_pred]).to(device)

		logits = model(Xr)
		if norm == 'l1':
			loss_r = loss_fn(logits, yf) + lamda * r.abs().sum()
		elif norm == 'l2':
			loss_r = loss_fn(logits, yf) + lamda * torch.linalg.norm(r) 

		optimizer_r.zero_grad()
		loss_r.backward(retain_graph=True)
		optimizer_r.step()

		if (i+1) % 1000 == 0: 
			sfmx = softmax(logits, dim=1).detach().cpu().numpy()[0]
			prob = sfmx[false_pred]
			# print(f'Prob (true: {y}, false: {false_pred}): {prob:>4f}')
			# print(f'Norm(r) = {torch.linalg.norm(r)}')
   
	if fig_file: 
		show_image(Xr.cpu().detach(), fig_file)  

	return float(torch.linalg.norm(r).detach().cpu().numpy()), prob 
