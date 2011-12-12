#! /opt/local/bin/python2.7
import math,random,numpy as np, png, time
from PIL import Image
import retina as r # TEMP: for debuggin only

''' cortex.py: contains all objects and functions necessary for simulating a neural network (cortex)
classes
	cortex: represents an entire neural network
	layer: represents a neural layer
	connection: represents an entire connection between two layers

usage
	# build the network
	s = layer(50,50,"input")
	d = layer(50,50,"output")
	con = connection(s,d,None) # connect the two layers
	network = cortex([s,d],'Neural Network')
	
	# run an update/learning cycle
	network.update_activations()
	network.update_weights()

dependencies
	numpy, PIL, PyPNG
		
notes
	(height,width)
	(dest,source)
	speed + memory usage is O(# connections)
	# connections is O(dest_neurons*conn_dest_neuron)
	np.array_str(the_np_array,precision=5,suppress_small=True)
	
'''

######### Params #########
SHOW_GUI = False
IS_VISUAL_CORTEX = False
k_kwta = 0.08#0.15
ENHANCE_CONTRAST = True
kWTA_reduction = 0.0
max_connections = 1000
max_cycles = 25# was 15, before that 100, but lets inf loops go too long
settle_threshold = 0.001 # was 0.001
# XCAL exponential decaying average
alpha_plus = 0.6 #0.6 std, 0.5 for v1
alpha_plus_theta = 0.0 # 0.0 std, I tried 0.05 as a hack
alpha_minus = 0.05 # 0.05 std
# XCAL learning rate
eta = 0.1 # std 0.1
xcal_thresh_offset = 0.0# 0.0 std, I tried 0.1 as a hack
# Activation Update Rate
dt = 0.8

print "--- Cortex.py ---"
print "eta:%.4f, alpha+:%.4f, alpha-:%.4f"%(eta, alpha_plus,alpha_minus)

##########################

# test
def init_testing_shit():
	r.cv.NamedWindow('Retina +')
	r.cv.NamedWindow('Retina -')
	r.cv.NamedWindow('V1')
	
	if IS_VISUAL_CORTEX:
		r.cv.NamedWindow('V4')
		r.cv.NamedWindow('IT')
		r.cv.MoveWindow('Retina +',0,0)
		r.cv.MoveWindow('Retina -',360,0) # was 360,0
		r.cv.MoveWindow('V1',0,400)
		r.cv.MoveWindow('V4',168,400)
		r.cv.MoveWindow('IT',168 + 240,400)
	else:
		r.cv.MoveWindow('Retina +',0,0)
		r.cv.MoveWindow('Retina -',201,0) # was 360,0
		r.cv.MoveWindow('V1',0,160)
	

class Cortex(object):
	''' cortex: a class representing a neural network
	Keeps track of layers involved, and provides easy methods for
	updating all activations and weights
	instance vars
		layers: a list of layer objects
		name: name of network
	'''
	def __init__(self,l,c,n):
		''' init(): initialize the network 
		inputs
			l: list of layer objects, in order of activations
			c: list of connection objects
			n: name of network
		outputs
			none
		effects
			stores l and n in instance vars
		'''
		self.layers = l
		self.connections = c
		self.name = n
		
		self.settle_cnt = 0
		if SHOW_GUI:
			init_testing_shit() # test
	
	def settle(self):
		''' settle(): run update cycles until the network settles
		inputs
			none
		outputs
			int: number of cycles to settle
		effects
			updates the activations and weights for every layer in the network
		'''
		[x.reset_act() for x in self.layers]
		
		if SHOW_GUI:
			retinal_layer = self.get_layer('retina')
			v1_layer = self.get_layer('v1')
			v4_layer = self.get_layer('v4')
			it_layer = self.get_layer('it')
			output_layer = self.get_layer('output')
		
		self.settle_cnt += 1
		loop_counter = max_cycles # max possible cycles
		total_wt_change = 0
		if SHOW_GUI:
			r.show_np('Retina +',retinal_layer.pos_act,retinal_layer.width, retinal_layer.height)
			r.show_np('Retina -',retinal_layer.neg_act,retinal_layer.width, retinal_layer.height)
			r.cv.WaitKey(10)
		
		while loop_counter > 0:
			loop_counter -= 1
			
			change = self.update_activations()
			
			total_wt_change += self.update_weights()
			
			
			if SHOW_GUI:
				r.show_np('V1',v1_layer.activations,v1_layer.width, v1_layer.height)
				if IS_VISUAL_CORTEX:
					r.show_np('V4',v4_layer.activations,v4_layer.width, v4_layer.height)
					r.show_np('IT',it_layer.activations,it_layer.width, it_layer.height)
#					r.show_np('Output',output_layer.activations,output_layer.width, output_layer.height)
				r.cv.WaitKey(50)
			
			if change < settle_threshold:
				break
		if SHOW_GUI:
			print "new settling cycle:%d\t%d\ttotal wt chng:%.3f"%(self.settle_cnt,max_cycles-loop_counter,total_wt_change)
		return max_cycles-loop_counter,total_wt_change
	
	def update_activations(self):
		''' update_activations(): update activations of all layers in network
		inputs
			none
		outputs
			sum of activation change
		effects
			updates the activations for every layer in the network via the
				calc_activation() method
		'''
		changes = [x.calc_activation() for x in self.layers]
		return sum(changes)
	
	def update_weights(self):
		''' update_activations(): update activations of all layers in network
		inputs
			none
		outputs
			sum of weight change
		effects
			updates the weights for every layer in the network via the do_learning()
				method
		'''
		return sum([x.do_learning() for x in self.layers])
	
	def get_layer(self,layer_name):
		''' get_layer(): gets the layer in this network with the given name
		inputs
			layer_name: name of the layer to get
		outputs
			the layer with the given name
		effects
			none
		'''
		for layer in self.layers:
			if layer.name.lower() == layer_name.lower():
				return layer
		return None
	
	def get_connection(self, conn_name):
		''' get_connection(): gets the connection in this network with the given name
		inputs
			conn_name: name of the connection to get
		outputs
			the connection with the given name
		effects
			none
		'''
		for connection in self.connections:
			if connection.name == conn_name:
				return connection
		return None
	
	def save_data(self,filename=None):
		''' save_data(): save all connections to disk
		inputs
			filename: (optional) string for path to file
		outputs
			none
		effects
			saves all data for each connection to ../data/weights
		'''
		print 'Saving data for',self.name
		[x.save(filename) for x in self.connections]
	
	def load_data(self,filename=None):
		''' load_data(): load data for all connections from disk
		inputs
			filename: (optional) string for path to file
		outputs
			none
		effects
			loads data from ../data/weights
			modifies every connection object based on these weights
		'''
		print 'Loading data for',self.name
		[x.load(filename) for x in self.connections]
	
	def print_conn_wt_stats(self):
		''' print_conn_wt_stats(): print stats for all connections
		inputs and outputs
			none
		effects
			prints to console
		'''
		for conn in self.connections:
			print_stats(conn.name,conn.weights)
	
	def print_structure(self):
		''' print_structure(): prints structure of self
		inputs and outputs
			none
		effects
			prints to console
		'''
		print "Name of Cortex:",self.name
		print "Layers:"
		for l in self.layers:
			print "\t",l.name,"(%d,%d)"%(l.width,l.height)
		print "Connections:"
		for c in self.connections:
			print "\t",c.name,"(%d,%d)"%(c.conn_box_width,c.conn_box_height)
	
class Layer(object):
	''' Layer: a class representing a neural layer.
	This keeps track of its size, connections to other neural layers (including
	itself), and its activations.
	instance vars
		width, height: floats defining physical dimensions of the layer
		size: width*height, size of numpy arrays 
		activations: 1-D numpy array representing activations of each neuron
		connections: list of connections to this neural layer
	'''
	def __init__(self,w,h,n,k=k_kwta):
		''' init(): initialize layer
		inputs
			w: float representing physical width of layer
			h: float representing physical height of layer
			n: string for name of this layer
			k: float (in [0,1]) defining amount of activation allowed
		outputs
			none
		effects
			sets width,height,size
			allocates activations
			inits connections
		'''
		self.name = n
		self.width = w
		self.height = h 
		self.size = w*h
		self.k_kwta = k
		self.clamped = False
		
		self.num_acts = np.zeros(self.size)
		self.activations = np.zeros(self.size)
		self.act_l = np.zeros(self.size)
		self.act_l.fill(0.5)
		self.connections = []
	
	def __str__(self):
		return "Layer %s: %dx%d=%d neurons"%(self.name,self.width,self.height,self.size)
	
	def save_image_for_activations(self,filename):
		''' save_image_for_activations(): save activations as an image
		inputs
			filename: string for name of file
		outputs
			none
		effects
			saves an image for num_acts
		'''
		image = np.zeros((self.width,self.height))
		for y in range(0,self.height):
			for x in range(0,self.width):
				image[y,x] = self.num_acts[y*self.width + x]
		image = image/np.max(image)
		im = Image.fromarray(np.uint8(image*255))
		im.save(filename)
		
	def connect(self,conn):
		''' connect(): add a connection to this neural layer
		inputs
			conn: Connection object representing a connection from another 
					neural layer to this one
		outputs
			none
		effects
			modifies connections to contain conn
		'''
		self.connections.append(conn)
	
	def reset_act(self):
		''' reset_act(): reset activations for cycle
		effects
			sets act to 0
		'''
		if not self.clamped:
			self.activations = np.zeros(self.activations.shape)
	
	def get_moving_average(self,act,act_l):
		''' get_moving_average() calculate updated averages
		inputs
			act: instantaneous activations, as numpy array
			act_l: long-term avg activations, as numpy array
		output
			updated long-term avg activations, as numpy array
		effects
			none
		'''
		delta = act-act_l
		condlist = [act>=act_l, act < act_l]
		choicelist = [alpha_plus*delta + alpha_plus_theta, alpha_minus*delta ]
		change = np.select(condlist,choicelist)
		act_l = act_l + change
		return act_l
		
	def calc_activation(self):
		''' calc_activation(): calculate activation of each neuron in this layer
		algorithm
			calculate strength of each connection
			sum for each neuron in this layer
			do sigma function
			set activations based on kWTA
		output
			sum of absolute value of change in activation this time
		effects
			updates self.activations
		'''
		if len(self.connections) == 0:
			return 0
		
		if not self.clamped:
			net_i = np.sum(np.vstack([x.get_net_i() for x in self.connections]), axis=0)
			net_i = net_i/len(self.connections)
			
			# new formula for calculating activations
			thresh = self.kWTA(net_i)
			y = net_i - thresh
			y = np.maximum(y,np.zeros(y.shape))
			new_act = self.amplify_activation(y)
#			if self.name == 'V1':
#				print_stats('net_i',net_i)
			old_act = self.activations
			self.activations = old_act + dt*(new_act - old_act)
			
#			if self.name == 'V1':
#				print_stats('V1 act',self.activations)
			
			# old formula for calculating activations
	#		vsigma = np.vectorize(sigma)
	#		sigma_act = vsigma(net_i)
	#		old_act = self.activations
	#		new_act = old_act + dt*(sigma_act - old_act) 
	#		self.activations = self.kWTA(new_act)
		else:
			old_act = self.activations
			
		# update long term activation
		self.act_l = self.get_moving_average(self.activations,self.act_l)
		
		self.num_acts = self.num_acts + self.activations # for testing, no effect
		return np.average(np.fabs(self.activations - old_act))
	
	def amplify_activation(self,y):
		''' amplify_activation(): do fxx1 on net_i-threshold to increase contrast
		input
			y: numpy array representing net_i-threshold
		output
			numpy array of same shape, amplified
		effects
			none
		'''
		fxx1V = np.vectorize(make_fxx1(100.0)) 
		return fxx1V(y)
	
	def kWTA(self,input_act):
		''' kWTA(): run kWTA inhibition on the input activations
		inputs
			input_act: 1d numpy array of activations
		output
			float for threshold value
		'''
		k = k_kwta*input_act.shape[0] # scale k with size of input
		sorted = np.sort(input_act)
		splits = np.split(sorted,[sorted.size - k])
		bottom_avg = np.mean(splits[0])
		top_avg = np.mean(splits[1])
		
		q = 0.5
		threshold = bottom_avg + q*(top_avg - bottom_avg) # threshold is a_theta
#		output_act = np.select([input_act>threshold,input_act<=threshold],[input_act,input_act*kWTA_reduction])
#		return output_act
		return threshold
	
	def normalizedInhibition(self,input_act):
		''' normalizedInhibition(): run normalized inhibition on 
			the input activations
		inputs
			input_act: 1d numpy array of activations
		output
			1d numpy array of inhibited activations
		'''
		goal_average = 0.5
		average_act = np.average(input_act)
		output_act = input_act*goal_average/average_act
		return output_act
		
	def do_learning(self):
		''' do_learning() update weights of each connection
		algorithm
			call do_XCAL() on each connection
			calculate total change
			(delegates learning to connection objects for greater flexibility)
		output
			returns float describing total change, to determine whether it has settled
		effects
			connection weights possibly changed
			act_l updated
		'''
		if len(self.connections) == 0:
			return 0.0
		total_change = sum(x.do_XCAL() for x in self.connections)
		return total_change
	
	def override_activations(self,val):
		''' override_activations(): set all activations to a certain value
		inputs
			val: float value to set for each activation
		'''
		self.activations.fill(val)


class CenterThresholdLayer(Layer):
	''' CenterThresholdLayer: represents a layer which calculates the kWTA 
	threshold with only the center 14*14 neurons
	'''
	def kWTA(self,input_act):
		''' kWTA(): run kWTA inhibition on the input activations
		use only the center 14*14 pixels
		inputs
			input_act: 1d numpy array of activations
		output
			float for threshold value
		'''
		w = self.width
		h = self.height
		x = (w-14)/2
		y = (h-14)/2
		x_vals = np.tile(np.arange(x,x+14), (14,1))
		y_vals = y + np.arange(14).reshape(14,1)
		idxs = x_vals + y_vals*w
		center_input = np.ravel(input_act[idxs])
		k = k_kwta*center_input.size # scale k with size of input
		sorted = np.sort(center_input)
		splits = np.split(sorted,[sorted.size - k])
		bottom_avg = np.mean(splits[0])
		top_avg = np.mean(splits[1])
		
		q = 0.5
		threshold = bottom_avg + q*(top_avg - bottom_avg)
		print "thresh:%.3f, %.3f->%.3f"%(threshold,bottom_avg,top_avg)
		return threshold


class InputLayer(Layer):
	''' InputLayer(): acts as an input layer. 
	'''
	def calc_activation(self):
		''' calc_activation(): no need for calculation on input
		effects
			none
		outputs
			sum of absolute value of change in activation this time (always 0)
		'''
		return 0
	
	def reset_act(self):
		''' reset_act(): DONT reset activations for input layer
		effects
			none
		'''
		pass


class OutputLayer(Layer):
	''' OutputLayer: a class representing an output of a neural network.
	Only allows one neuron to be active at a time
	Also allows clamping output
	'''
	def kWTA(self,input_act):
		''' kWTA(): only allow one neuron to be active at a time
		inputs
			input_act: 1d numpy array of activations
		output
			float for threshold value
		'''
		k = k_kwta*input_act.shape[0] # scale k with size of input
		sorted = np.sort(input_act)
		thresh = (sorted[-1] + sorted[-2])/2.0
		return thresh
	
	def amplify_activation(self,y):
		''' amplify_activation(): do fxx1 on net_i-threshold to increase contrast
		input
			y: numpy array representing net_i-threshold
		output
			numpy array of same shape, amplified
		effects
			none
		'''
		fxx1V = np.vectorize(make_fxx1(2000.0)) 
		return fxx1V(y)
	
	def clamp(self,input):
		''' clamp(): clamp output for a certain neuron or set of neurons
		inputs
			input: index of neuron to clamp or list of indexes to clamp on
		output
			sets all elements of self.activations from input to 1.0, everything else to 0
			sets clamped to True
		'''
		self.clamped = True
		self.activations.fill(0.0)
		try:
			for i in input:
				self.activations[i] = 1.0
		except TypeError:
			self.activations[input] = 1.0
	
	def unclamp(self):
		''' unclamp(): unclamp output
		inputs
			none
		output
			sets clamped to False
		'''
		self.clamped = False
	
	def get_answer(self):
		''' get_answer(): get the output from this layer
		inputs
			none
		output
			index of answer (largest activation value)
		'''
		return np.argmax(self.activations)

class Connection(object):
	''' connection: a class representing a connection between two neural layers
	instance vars
		source, dest: layer objects
		source_idxs = connections between source and destination neurons
				dimensions are dest.size*n where n is number of connections/destination neuron
				source_idxs holds all indexes of source neurons, where each row is one destination neuron
		weights = dest.size*n, each row corresponds to a destination neuron, and 
				each weight corresponds to the source neuron given by the index 
				in conns at the same location
	'''
	def __init__(self,s,d,size=None,inhib=False,strength=1.0,fixed=False):
		''' init(): initialize connection between layers
		inputs
			s: source layer (a layer object)
			d: destination layer (a layer object)
			size: (width,height) of 'connection box', or None for default
			inhib: bool defining whether this connection is inhibitory or excitatory
			strength: float for strength of the connection (this is multiplied by net_i)
		effects
			set source, dest,filename,name
			allocate source_idxs, weights
			inits weights
		'''
		self.fixed_weights = fixed
		self.inhibitory = inhib
		self.strength = strength
		self.source = s
		self.dest = d
		self.set_name()
		self.filename = '../data/weights/' + self.name
		
		if size == None:
			c_per_dest = max_connections
			if c_per_dest > s.size:
				c_per_dest = s.size
			self.conn_box_width = int(math.sqrt(c_per_dest))
			self.conn_box_height = int(c_per_dest/self.conn_box_width)
		else:
			self.conn_box_width = size[0]
			self.conn_box_height = size[1]
			
		self.conn_per_dest = self.conn_box_width*self.conn_box_height
		
		self.source_idxs = np.zeros((d.size,self.conn_per_dest), dtype=np.int32)
		self.weights = np.zeros((d.size,self.conn_per_dest), dtype=np.float32)
		
		self.init_weights()
		self.dest.connect(self)
	
	def set_name(self):
		''' set_name(): set the name of this connection
		effects
			updates self.name and 
		'''
		self.name = self.source.name + "_" + self.dest.name
			
	def save_as_image(self,filename):
		''' save_as_image(): save weights as an image to disk, for human viewing
		inputs
			filename: string for path of image
		outputs
			none
		effects
			none
		'''
#		print "saving image ...",
		box_w,box_h = self.conn_box_width,self.conn_box_height
		dest_w,dest_h = self.dest.width,self.dest.height
		im_w = dest_w*(box_w+1)
		im_h = dest_h*(box_h+1)
		image = np.zeros((im_w,im_h))
		wts = self.get_weights()
		print_stats('wts',self.weights)
		print_stats('contrast enhanced wts',wts)
		for dest_x in range(dest_w):
			for dest_y in range(dest_h):
				x_o = dest_x*(box_w+1)
				y_o = dest_y*(box_h+1)
				i = dest_y*dest_w + dest_x
				# draw weights from here
				indexes = np.zeros(box_w*box_h, dtype=np.int32)
				for box_x in range(box_w):
					for box_y in range(box_h):
						box_i = box_y*box_w + box_x
						x_out = x_o + box_x
						y_out = y_o+box_y
						image[y_out,x_out] = wts[i][box_i]
		
		file = open(filename,'wb')
		w = png.Writer(im_w,im_h,greyscale=True)
		w.write(file, image*255.0)
		file.close()
		print "saved image to", filename
	
	def save(self,filename=None):
		''' save(): save this connection to disk
		inputs
			filename (optional) string for path to file
		outputs
			none
		effects
			saves source_idxs,weights
		notes
			all other data must be set properly. this is not robust, so if you 
				load bad weights, stuff will break
		'''
		if filename == None:
			filename = self.filename
		else:
			filename += self.name
		
		np.save(filename + '__source_idxs.npy',self.source_idxs)
		np.save(filename + '__weights.npy',self.weights)
	
	def load(self,filename=None):
		''' load(): load data for this connection from disk
		inputs
			filename (optional) string for path to file
		outputs
			none
		effects
			loads source_idxs,weights from self.filename
		notes
			all other data must be set properly. this is not robust, so if you 
				load bad weights, stuff will break
		'''
		if filename == None:
			filename = self.filename
		else:
			filename += self.name
		
		try:
			self.source_idxs = np.load(filename + '__source_idxs.npy')
			self.weights = np.load(filename + '__weights.npy')
		except IOError:
			pass
	
	def init_weights(self):
		''' initWeights(): set the weights between the two neural networks
		To make a non-random connection scheme, override this function
		To make fixed weights, override this function and set fixed_weights to True
		algorithm:
			sets all weights to a random value. Connects each dest neuron to 
				the conn_per_dest neurons closest to it
		effects
			initializes values in self.weights
		'''
		s_w = self.source.width
		s_h = self.source.height
		box_w = self.conn_box_width
		box_h = self.conn_box_height
		for i in xrange(self.dest.size):
			# make a box in source closest to dest neuron at (x,y)
			# fill each weight with rand_weight()
			y = i/self.dest.width
			x = i - y*self.dest.width
			x_o = int(x*float(s_w)/float(self.dest.width)) - box_w/2
			y_o = int(y*float(s_h)/float(self.dest.height)) - box_h/2
			if x_o + box_w > s_w:
				x_o = s_w - box_w
			if y_o + box_h > s_h:
				y_o = s_h - box_h
			if x_o < 0:
				x_o = 0
			if y_o < 0:
				y_o = 0
			if x_o + box_w > s_w:
				box_w = s_w - x_o
			if y_o + box_h > s_h:
				box_h = s_h - y_o
			
			source_row = self.source_idxs[i]
			
			# numpy magic (15sec, confusing)
#			x_vals = np.tile(np.arange(x_o, x_o + box_w),box_h)
#			y_vals = np.repeat(np.arange(y_o, y_o + box_h),box_w)
#			source_row[:] = x_vals+s_w*y_vals
			
			# pure python (15sec, readable)
			s_index = 0
			for s_y in range(y_o, y_o + box_h):
				for s_x in range(x_o, x_o + box_w):
					source_row[s_index] = s_x + s_y*s_w
					s_index += 1
					
		self.weights = np.random.normal(0.50,0.141421,self.weights.shape) # avg (std 0.5) stddev (std 0.141421)
#		gauss_sigma = 0.707106781186548
		#self.weights = np.fabs(gauss_sigma*np.random.standard_normal(self.weights.shape))
		self.weights = np.minimum(self.weights,np.ones(self.weights.shape)) # cap at 1
		self.weights = np.maximum(self.weights,np.zeros(self.weights.shape)) # cap at 0
	
	def source_activations(self):
		''' source_activations(): get source activations
			This is here so you can override it.
		output
			self.source.activations, a 1d numpy array
		'''
		return self.source.activations
	
	def get_connections_per_neuron(self):
		''' get_connections_per_neuron(): return the number of connections per 
			destination neuron
		output
			int with number of connections
		effects
			none
		'''
		return self.conn_per_dest
	
	def get_weights(self):
		''' get_weights(): get weights used for calculating net_i
			this uses a contrast enhancement function
		outputs
			numpy array of weights, with enhanced contrast
		effects
			none
		'''
		w = self.weights
		if ENHANCE_CONTRAST:
			theta = 1.0
			gain = 6.0#std 6.0, 3.0 is for testing
			w = 1 / (1 + (w/(theta*(1.001-w)))**(-gain))
		return w
	
	def get_net_i(self):
		''' get_net_i(): calculate net_i for this set of connections
		output
			numpy array (1*dest.size) with total input to each destination neuron
		'''
		acts = self.source_activations()[self.source_idxs]
		conn_strs = acts*self.get_weights()
		sum_i = np.sum(conn_strs,axis=1)
		net_i = sum_i/self.get_connections_per_neuron()
		if self.inhibitory:
			net_i = -net_i
		net_i = net_i*self.strength
#		print "net_i:%.3f"%np.sum(net_i)
		return net_i
	
	def do_XCAL(self):
		''' do_XCAL(): perform XCAL learning algorithm on all connections
		algorithm
			get xy averages
			calculate necessary change in weight using xcal function
			modify weights based on learning rate
			update xy_m averages
		output
			float describing total change in weights
		effects
			weights and xy_m changed for each connection
		'''
		total_change = 0.0
		if not self.fixed_weights:
			
			x = s_acts = self.source_activations()[self.source_idxs]
			
			d_acts = self.dest.activations
			d_acts_wide = np.tile(d_acts,(s_acts.shape[1],1))
			d_acts_wide_t = np.swapaxes(d_acts_wide,0,1)
			y = d_acts_wide_t
			
			d_act_l = self.dest.act_l
			d_act_l_wide = np.tile(d_act_l,(s_acts.shape[1],1))
			d_act_l_wide_t = np.swapaxes(d_act_l_wide,0,1)
			y_l = d_act_l_wide_t
			xy = x*y
			xy_l = x*y_l
			
			deltaW_xy = f_xcal(xy,xy_l)
			dwt = eta*deltaW_xy
			wt = self.weights
			condlist = [dwt > 0, dwt <= 0]
			choicelist = [(1 - wt) * dwt, wt*dwt]
			changes = np.select(condlist,choicelist) 

			
			self.weights = self.weights + changes
			
#			print "wt chng: avg wt %.3f, avg chng %.3f"%(np.average(np.sum(self.weights, axis=1)),np.average(np.sum(changes, axis=1)))
#			print_array(np.sum(changes, axis=1)[self.dest.activations > 0.1])
#			print "sum of wts: avg is %.3f"%np.average(np.sum(self.weights, axis=1))
#			print_array(np.sum(self.weights, axis=1)[self.dest.activations > 0.1])
#			print "sum of contrast-enhanced wts: avg is %.3f"%np.average(np.sum(self.get_weights(), axis=1))
#			print_array(np.sum(self.get_weights(), axis=1)[self.dest.activations > 0.1])
			
			total_change = np.sum(changes)#np.fabs(changes))
			
		return total_change


class FixedMountainConnection(Connection):
	''' FixedMountainConnection: special type of connection, where the weights are
		highest near the destination neuron, and progressively decrease going outwards
	instance vars
		source, dest: layer objects
		source_idxs = connections between source and destination neurons
				dimensions are dest.size*n where n is number of connections/destination neuron
				source_idxs holds all indexes of source neurons, where each row is one destination neuron
		weights = dest.size*n, each row corresponds to a destination neuron, and 
				each weight corresponds to the source neuron given by the index 
				in conns at the same location
	'''
	def init_weights(self):
		''' initWeights(): set the weights between the two neural networks
		algorithm
			this connects a neuron to its neighboring neurons based on distance
			all weights are fixed
		effects
			initializes values in self.weights
		'''
		self.fixed_weights = True
		s_w = self.source.width
		s_h = self.source.height
		box_w = self.conn_box_width
		box_h = self.conn_box_height
		for i in xrange(self.dest.size):
			y = i/self.dest.width
			x = i - y*self.dest.width
			
			source_row = self.source_idxs[i]
			s_index = 0
			for delta_y in range(-box_h/2+1, box_h/2+1):
				for delta_x in range(-box_w/2+1,box_w/2+1):
					s_y = y+delta_y
					s_x = x+delta_x
					if s_y >= s_h:
						s_y -= s_h
					if s_x >= s_w:
						s_x -= s_w
					if s_y <0:
						s_y += s_h
					if s_x <0:
						s_x += s_w
					source_row[s_index] = s_x + s_y*s_w
					s_index += 1
					
		mtn_array = np.ravel(mountain_array(min(self.conn_box_width,self.conn_box_width)))
		self.weights = np.tile(mtn_array,(self.dest.size,1))


def mountain_array(size):
	''' mountain_array(): build weights for v1-v1, looks like a mountain
	inputs
		size: int dictating size of each dimension
	output
		size*size numpy array of floats
	effects
		none
	'''
	if size % 2 == 0:
		print "Error in mountain_array(): size %d is even"%size
		return None
	
	center = (size-1)/2
	offset = math.sqrt((center)**2 + (center)**2) # val of corners
	
	array = np.zeros((size,size))
	for x in range(size):
		for y in range(size):
			if not x == center or not y == center:
				array[x,y] = offset-math.sqrt((x-center)**2 + (y-center)**2)
	desired_avg = 0.5
	array = array * desired_avg/np.average(array)		
	return array

def rand_weight():
	''' generate random weight with gaussian distribution
	not used anymore
	'''
	return math.fabs(random.gauss(0,0.5))

def sigma(input):
	''' do sigma on input to cap values in [-1.0,1.0]
	'''
	return 1.0/(1.0+math.exp(-input))


def make_fxx1(gain):
	''' make_fxx1(): make a function that runs fxx1 with arbitrary gain
	'''
	def fxx1_internal(input):
		''' do fxx1 on input
		'''
		return gain*input/(gain*input+1)
	return fxx1_internal
	
#def fxx1(input):
#	gain = 100.0
#	return gain*input/(gain*input+1)

def f_xcal(a,b):
	''' f_xcal() do XCAL calculation
	inputs
		a,b: numpy arrays of the same size (typically xy,xy_l)
	outputs:
		numpy array of same size
	'''
	condlist = [a>.1*b,a <= .1*b]
	choicelist = [a-b,-9*a]
	return np.select(condlist,choicelist)
	
start = time.time()
def start_timer():
	global start
	start = time.time()

def print_timer(msg):
	global start,next
	next = time.time()
	print '%.4fs' % (next - start),msg
	start = next

def print_stats(name, input):
	vals = (np.max(input),np.min(input),np.average(input),np.std(input))
	print "max:%.3f, min:%.3f, avg:%.3f, std:%.3f <-- " % vals, name

def print_array(arr):
	print np.array_str(arr,precision=4,suppress_small=True)
	
def test():
	s = Layer(4,4,"bottom")
	d = Layer(4,4,"top")
	con = Connection(s,d,None)
	print "Layer:",str(s)
	print "Layer:",str(d)
	print "Weights:\n",con.weights
	
	s.override_activations(1.0)
	d.calc_activation()
	d.do_learning()
