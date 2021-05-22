import pickle

with open('weights.pickle','rb') as handle:
	weights = pickle.load(handle)
handle.close()

with open('biases.pickle','rb') as handle:
	biases = pickle.load(handle)
handle.close()

with open('layers.pickle','rb') as handle:
	layers = pickle.load(handle)
handle.close()

print('\n')
total_param = 0
for layer in layers:
	print(layer)
	print('Weights: ', weights[layer].shape)
	print('Biases : ', biases[layer].shape)
	params = weights[layer].size + biases[layer].size
	print('Layer params :', params)
	total_param += params
	print('\n')
	
print('Total parameters: ', total_param)
print('\n')
