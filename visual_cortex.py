#! /opt/local/bin/python2.7
import cortex as c, numpy as np, math, retina as r,os,time
from PIL import Image

''' visual_cortex.py: this is an attempt to model the human visual cortex
classes
	RetinalLayer: subclass of cortex.Layer, representing a Retina
	RetinaV1PositiveConnection,RetinaV1NegativeConnection: subclasses of 
		cortex.Connection, representing connections between the retina and V1 layers

usage
import visual_cortex as v; v.retina_test()



dependencies
	cortex.py, retina.py, numpy, PIL, OpenCV (via retina.py)

'''
		
class RetinalLayer(c.InputLayer):
	def __init__(self,w,h,n):
		''' init(): initialize layer
		inputs
			w: float representing physical width of layer
			h: float representing physical height of layer
			n: string for name of this layer
		outputs
			none
		effects
			sets width,height,size
			allocates activations
			inits connections
		'''
		super(RetinalLayer,self).__init__(w,h,n)
		self.pos_act_l = np.zeros(self.size)
		self.neg_act_l = np.zeros(self.size)
		self.pos_act = np.zeros(self.size)
		self.neg_act = np.zeros(self.size)
		
	def load_image(self,filename,detailed=False,random=False):
		''' load_image() load an image and set activations
		inputs
			filename: path to file
		outputs
			none
		effects
			sets activations (in range [-1.0,1.0])
			sets pos_act (activations capped at [0.0,1.0])
			sets neg_act (-activations capped at [0.0,1.0])
		'''
		ganglion = r.calc_retinal_layer(filename,self.width,self.height,detailed=detailed,is_random=random)
		self.activations = np.ravel(ganglion)
		self.pos_act = np.maximum(self.activations,np.zeros(self.activations.shape))
		self.neg_act = np.maximum(-self.activations,np.zeros(self.activations.shape))


class RetinaV1Connection(c.Connection):
	''' RetinaV1Connection: class representing retina->V1 connections
	Specifically, this handles the block connection scheme
	'''
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
		self.dest_block_size = 14*14
		self.dest_block_width = 14
		self.dest_block_height = 14
		
		self.fixed_weights = False
		s_w = self.source.width
		s_h = self.source.height
		s_box_w = self.conn_box_width
		s_box_h = self.conn_box_height
		for i in xrange(self.dest.size):
			# get x,y location of this neuron
			y = i/self.dest.width
			x = i - y*self.dest.width
			
			# get corresponding location in source layer
			s_x = int(x*float(s_w)/float(self.dest.width))
			s_y = int(y*float(s_h)/float(self.dest.height))

			# round s_x,s_y to nearest source box
			x_o = s_x - s_x%(s_box_w/4)
			y_o = s_y - s_y%(s_box_h/4)
			
			if x_o + s_box_w > s_w:
				x_o = s_w - s_box_w
			if y_o + s_box_h > s_h:
				y_o = s_h - s_box_h
			if x_o < 0:
				x_o = 0
			if y_o < 0:
				y_o = 0
			if x_o + s_box_w > s_w:
				s_box_w = s_w - x_o
			if y_o + s_box_h > s_h:
				s_box_h = s_h - y_o
			
			source_row = self.source_idxs[i]
			
			# pure python (15sec, readable)
			s_index = 0
			for s_y in range(y_o, y_o + s_box_h):
				for s_x in range(x_o, x_o + s_box_w):
					source_row[s_index] = s_x + s_y*s_w
					s_index += 1
					
		self.weights = np.random.normal(0.50,0.141421,self.weights.shape) # avg (std 0.5) stddev (std 0.141421)
		self.weights = np.minimum(self.weights,np.ones(self.weights.shape)) # cap at 1
		self.weights = np.maximum(self.weights,np.zeros(self.weights.shape)) # cap at 0
	
	def load(self,filename=None):
		''' load(): load data for this connection from disk
		inputs
			filename (optional) string for path to file
		outputs
			none
		effects
			loads source_idxs,weights from self.filename
		notes
			This will also load a smaller set of weights and replicate it many times, if necessary
			It is assumed that the smaller set is the same size as one connection block in destination
			This only loads source indexes if the file is the same shape as this connection
		'''
		if filename == None:
			filename = self.filename
		else:
			filename += self.name
		
		try:
			new_sources = np.load(filename + '__source_idxs.npy')
			new_weights = np.load(filename + '__weights.npy')
			if new_sources.shape == self.source_idxs.shape:
				self.source_idxs = new_sources
			
			if new_weights.shape == self.weights.shape:
				self.weights = new_weights
			elif new_weights.shape[0] == self.dest_block_size and new_weights.shape[1] == self.get_connections_per_neuron():
				self.weights = np.tile(new_weights,(self.weights.shape[0]/new_weights.shape[0],1))
				 
		except IOError:
			pass


class RetinaV1PositiveConnection(RetinaV1Connection):
	def source_activations(self):
		''' source_activations(): get source activations
			This is here so you can override it.
		'''
		return self.source.pos_act
	
	def set_name(self):
		''' set_name(): set the name of this connection
		effects
			updates self.name
		'''
		self.name = self.source.name + "_" + self.dest.name + '_p'
	
	def save_image_rel(self,filename, neg_conn):
		''' save_as_image(): save weights as an image to disk, for human viewing
		inputs
			filename: string for path of image
			neg_conn: RetinaV1NegativeConnection object
		outputs
			none
		effects
			none
		'''
		pos_wt,neg_wt = self.get_weights(),neg_conn.get_weights()
		box_w,box_h = self.conn_box_width,self.conn_box_height
		dest_w,dest_h = self.dest.width,self.dest.height
		im_w = dest_w*(box_w+1)
		im_h = dest_h*(box_h+1)
		image = np.zeros((im_w,im_h,3))
		scale = 2.0
		for dest_x in range(dest_w):
			for dest_y in range(dest_h):
				x_o = dest_x*(box_w+1)
				y_o = dest_y*(box_h+1)
				i = dest_y*dest_w + dest_x
				# draw weights from here
				for box_x in range(box_w):
					for box_y in range(box_h):
						box_i = box_y*box_w + box_x
						x_out,y_out = x_o + box_x,y_o+box_y
						pos,neg = pos_wt[i][box_i],neg_wt[i][box_i]
						diff = scale*(pos - neg)
						if math.fabs(diff) > 0.999:
							diff = diff / math.fabs(diff)
						
						if diff > 0:
							image[y_out,x_out,0] = diff
						else:
							image[y_out,x_out,2] = -diff
		
		im = Image.fromarray(np.uint8(image*255))
		im.save(filename)
		print "saved image to", filename

class RetinaV1NegativeConnection(RetinaV1Connection):
	def source_activations(self):
		''' source_activations(): get source activations
			This is here so you can override it.
		'''
		return self.source.neg_act
	
	def set_name(self):
		''' set_name(): set the name of this connection
		effects
			updates self.name
		'''
		self.name = self.source.name + "_" + self.dest.name + '_n'


################ Testing Functions ################ 

def build_v1_cortex():
	strengths = 1.0#0.526315789473684
	v1_strength = strengths*0.5#1.0/3.0
	v1_inhib = False
	print "strengths:%.5f"%strengths
	print "v1_strength:%.5f"%v1_strength
	print "v1_inhib:",v1_inhib
	print "--- Building Visual Cortex ---"
	c.start_timer()
	# retina.shape = (12,12)x
	# v1.shape = (14,14)x
	retina = RetinalLayer(12,12,'Retina') #emerg: 12x12
	v1 = c.Layer(14,14,'V1')#,k=0.3) #emerg: 14x14
	c.print_timer("made layers")
	
	ret_v1_p = RetinaV1PositiveConnection(retina,v1,(12,12),strength=strengths)
	ret_v1_n = RetinaV1NegativeConnection(retina,v1,(12,12),strength=strengths)
	v1_v1 = c.FixedMountainConnection(v1,v1,(11,11),inhib=v1_inhib,strength=v1_strength) #emerg: 11x11
	c.print_timer("made connections")
	
	network = c.Cortex([retina,v1],[ret_v1_p,ret_v1_n,v1_v1],'Visual Cortex')
	network.print_structure()
	network.load_data('../data/v1 tests/weights/')
	return network

def build_visual_cortex():
	strengths = 1.0
	v1_strength = strengths*0.5
	v1_inhib = False
	print "strengths:%.5f"%strengths
	print "v1_strength:%.5f"%v1_strength
	print "v1_inhib:",v1_inhib
	print "--- Building Visual Cortex ---"
	c.start_timer()
	# retina.shape = (12,12)x
	# v1.shape = (14,14)x
	retina = RetinalLayer(36,36,'Retina') #emerg: 12x12
	v1 = c.Layer(168,168,'V1')#emerg: 14x14
	v4 = c.Layer(24,24,'V4')
	it = c.OutputLayer(10,10,'IT')
	c.print_timer("made layers")
	
	ret_v1_p = RetinaV1PositiveConnection(retina,v1,(12,12),strength=strengths,fixed=True)
	ret_v1_n = RetinaV1NegativeConnection(retina,v1,(12,12),strength=strengths,fixed=True)
	v1_v1 = c.FixedMountainConnection(v1,v1,(7,7),inhib=v1_inhib,strength=v1_strength) #emerg: 11x11
	v1_v4 = c.Connection(v1,v4,(12,12))
	v4_it = c.Connection(v4,it,None)
	it_v4 = c.Connection(it,v4,None)
	c.print_timer("made connections")
	
	network = c.Cortex([retina,v1,v4,it],[ret_v1_p,ret_v1_n,v1_v4,v4_it,it_v4],'Visual Cortex')
	network.print_structure()
	network.load_data()
	return network


def io_test():
	network = build_cortex()
	ret_v1_p = network.get_connection('Retina_V1_p')
	ret_v1_n = network.get_connection('Retina_V1_n')
	v1_v1 = network.get_connection('V1_V1')
	
	print "--- BEFORE ---"
	c.print_array(ret_v1_n.source_idxs)
	c.print_array(ret_v1_n.weights)
	
	network.load_data() # tests error handling if no file
	network.save_data()
	network.load_data()
	
	print "--- MID ---"
	c.print_array(ret_v1_n.source_idxs)
	c.print_array(ret_v1_n.weights)
	
	network = None
	new_net = build_cortex()
	ret_v1_p = new_net.get_connection('Retina_V1_p')
	ret_v1_n = new_net.get_connection('Retina_V1_n')
	v1_v1 = new_net.get_connection('V1_V1')
	
	print "--- NEW ---",ret_v1_n.weights.shape
	c.print_array(ret_v1_n.source_idxs)
	c.print_array(ret_v1_n.weights)
	
	new_net.load_data()
	
	print "--- AFTER ---",ret_v1_n.weights.shape
	c.print_array(ret_v1_n.source_idxs)
	c.print_array(ret_v1_n.weights)

def obj_rec_train():
	network = build_visual_cortex_2()
	network.print_conn_wt_stats()
	input = network.get_layer('Gabor')
	output = network.get_layer('IT')
	out_file = open('../data/recent/accuracy.txt','a')
	start_time = time.time()
	
	base_path = '../images/numbers/'
	num_training_cycles = 200
	for i in range(num_training_cycles):
		# get list of images
		num_correct = 0
		paths = os.listdir(base_path)
		print "Training on images:",paths
		for path in paths:
			answer = int(path.split('.')[0])
			print 'training on',path,'=',answer,' unclamped phase ...',
			input.load_image(base_path + path)
			cycles_unclamped,wt_change = network.settle()
			guess = output.get_answer()
			print 'guess: %d. Now clamping ...'%guess,
			if guess == answer:
				num_correct += 1
			output.clamp(answer)
			cycles_clamped,wt_change = network.settle()
			output.unclamp()
			print '%d,%d'%(cycles_unclamped,cycles_clamped)
			
		accuracy = (float(num_correct)/float(len(paths)))
		print 'accuracy:%.3f\ttime:%.3f\titeration:%d'%(accuracy,time.time() - start_time,i+1)
		out_file.write('%.4f\n'%accuracy)
		out_file.flush()
		network.print_conn_wt_stats()
		network.save_data()

	print '-- Total elapsed time:%.3fsec'%(time.time() - start_time)
	out_file.close()

def retina_test():
	network = build_v1_cortex()
	ret = network.get_layer('Retina')
	c.print_timer("built layer")
	
	start_time = time.time()
	
	ret_v1_p = network.get_connection('Retina_V1_p')
	ret_v1_n = network.get_connection('Retina_V1_n')
	v1 = network.get_layer('v1')
	c.print_stats('pos wt',ret_v1_p.weights)
	c.print_stats('neg wt',ret_v1_n.weights)
	base_path = '../kyo/'
	for i in range(400): # go for 129
		if i%10 == 0:
			val = i
			ret_v1_p.save_image_rel('../data/v1 tests/ret_v1_'+str(val)+'.png',ret_v1_n)
			ret_v1_p.save_as_image('../data/v1 tests/ret_v1_p_'+str(val)+'.png')
			ret_v1_n.save_as_image('../data/v1 tests/ret_v1_n_'+str(val)+'.png')
			v1.save_image_for_activations('../data/v1 tests/v1_'+str(val)+'.png')
			network.print_conn_wt_stats()
		pass_time = time.time()
		listing = os.listdir(base_path)
		cycles = []
		total_wt_change = 0
		for idx,path in enumerate(listing):
			ret.load_image(base_path + path,random=True,detailed=False)
			cycle_cnt,wt_change = network.settle()
			total_wt_change += wt_change
			cycles.append(cycle_cnt)
		if i%50 == 0:
			network.save_data('../data/v1 tests/weights/')
		print '-- Pass %d: %.3fsec (%.2f cycles each for %d imgs)'%(i,time.time()-pass_time,float(sum(cycles))/len(cycles),len(listing))
				
	print '-- Total elapsed time:%.3fsec'%(time.time() - start_time)
	ret_v1_p.save_image_rel('../data/v1 tests/post_ret_v1.png',ret_v1_n)
	c.print_stats('pos wt',ret_v1_p.weights)
	c.print_stats('neg wt',ret_v1_n.weights)



if __name__ == '__main__':
    retina_test()

