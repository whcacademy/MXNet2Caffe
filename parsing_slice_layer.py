

import argparse
import re
import json


def arg_parse():
	parser = argparse.ArgumentParser(description='Argument Parser for mxnet2caffe model parsing')
	parser.add_argument('-f', '--filename', help='the name of the file which constain the id', 
							default='testlist.prototxt', type=str)
	args = parser.parse_args()
	return args

def refine(str_item):
	return str_item.replace("'", '"').replace(' u', ' ').replace('{u', '{')

def parse_file(filename):
	with open(filename, 'r') as reader:
		model = reader.read()

	json_strs = re.findall(r'<slice_json>\n(.+?)\n</slice_json>', model)
	return list(map(lambda x: json.loads(refine(x)), json_strs))

def construct_slice_layer(slice_candidates):
	layer_dict = {'bottom' : None, 'axis': None, 
					'tops' : [], 'split_point':[], 'name' : None}
	layer_dict['bottom'] = slice_candidates[0]['bottom'][0]
	layer_dict['name'] = 'slicer_'+layer_dict['bottom']
	layer_dict['axis'] = slice_candidates[0]['param']['axis']
	for i in range(len(slice_candidates)-1):
		layer_dict['split_point'].append(slice_candidates[i]['param']['end'])
		layer_dict['tops'].append(slice_candidates[i]['top'])
	layer_dict['tops'].append(slice_candidates[len(slice_candidates)-1]['top'])
	return layer_dict

def get_slice_layers(jsons):
	bottoms = set(list(map(lambda x: x['bottom'][0], jsons)))
	slice_candidates = [[y for y in jsons if y['bottom'][0]==x] for x in bottoms]
	return list(map(lambda x: construct_slice_layer(x),slice_candidates))

def write_slice_layer(slice_layers, bottom, writer):
	# find the slice layer with target bottom
	layers = list(filter(lambda x : x['bottom'] == bottom, slice_layers))
	if len(layers):
		layer = layers[0]
		write_one_slice_layer(writer, layer)
		slice_layers.remove(layer)
	else:
		pass
	return slice_layers

def replace_slice_json(filename, slice_layers):
	temp = filename.replace('.prototxt','') + '_sliced.prototxt'
	write_flag = True
	with open(filename, 'r') as reader:
		with open(temp, 'w') as writer:
			line = reader.readline()
			while line:
				if line == '<slice_json>\n':
					write_flag = False
				if write_flag:
					writer.write(line)
				else:
					# get the bottom info
					bottom = re.findall(r"'bottom': \['(.+?)'\]", line)
					if (len(bottom)):
						slice_layers = write_slice_layer(slice_layers, bottom[0], writer)
				if line == '</slice_json>\n':
					write_flag = True
				line = reader.readline()
	return temp

def write_one_slice_layer(writer, layer):
	writer.write('\n')
	writer.write('layer {\n')
	writer.write('  bottom: "%s"\n' % layer['bottom'])
	writer.write('  name: "%s"\n'   % layer['name'])
	writer.write('  type: "Slice"\n')
	for top in layer['tops']:
		writer.write('  top: "%s"\n' % top)
	writer.write('  slice_param {\n')
	writer.write('  	axis: %s\n' % layer['axis'])
	for split_point in layer['split_point']:
		writer.write('  	slice_point: %s\n' % split_point)
	writer.write('  }\n')
	writer.write('}\n')
	writer.write('\n')

# interface for in-python call
def handle_slice_layers(filename):
	jsons = parse_file(args.filename)
	slice_layers = get_slice_layers(jsons)
	newfile = replace_slice_json(args.filename, slice_layers)
	# append_slice_layer(newfile, slice_layers)
	return newfile

# independent call
if __name__ == '__main__':
	args = arg_parse()
	jsons = parse_file(args.filename)
	slice_layers = get_slice_layers(jsons)
	newfile = replace_slice_json(args.filename, slice_layers)
	# append_slice_layer(newfile, slice_layers)
