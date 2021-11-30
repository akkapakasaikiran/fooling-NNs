import argparse

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-d', '--data_dir', type=str, default='../data/')
	parser.add_argument('-m', '--models_dir', type=str, default='../models/')
	parser.add_argument('-f', '--figs_dir', type=str, default='../figs/')
	parser.add_argument('-r', '--results_dir', type=str, default='../results/')
	parser.add_argument('-c', '--checkpoint_dir', type=str, default='../models/')
	parser.add_argument('--cuda', type = int, default = '-1')

	args = parser.parse_args()
	return args