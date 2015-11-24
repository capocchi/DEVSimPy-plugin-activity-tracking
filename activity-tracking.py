# -*- coding: utf-8 -*-

"""
	Authors: L. Capocchi (capocchi@univ-corse.fr),
			J.F. Santucci (santucci@univ-corse.fr)
	Date: 12/09/2013
	Description:
		Activity tracking for DEVSimPy
		We add dynamically a 'activity' attribute to the Block at the GUI level and 'texec'
		(which is dico like {'fnc':[(t1,t1'),(t2,t2'),..]}
		where fnc is the selected transition function, t the simulation time
		(or number of event) and t' the execution time of fcnt.)
		attribute at the DEVS level. We deduct the tsim doing the sum of texec.
	Depends: 'python-psutil' for cpu usage, networkx and pylab for graph
			 codepaths file
"""

### ----------------------------------------------------------

### at the beginning to prevent with statement for python version <=2.5
from __future__ import with_statement

import sys
import wx
import wx.grid
import os
import inspect
import tempfile
import textwrap
import csv
import timeit #http://pymotw.com/2/timeit/
import functools
import random

# to send event
if wx.VERSION_STRING < '2.9':
	from wx.lib.pubsub import Publisher as pub
else:
	from wx.lib.pubsub import setuparg1
	from wx.lib.pubsub import pub

#for plotting
try:
	import pydot
except ImportError, info:
	platform_sys = os.name
	if platform_sys in ('nt', 'mac'):
		msg = _("ERROR: pydot module not found.\nhttp://code.google.com/p/pydot/\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	elif platform_sys == 'posix':
		msg = _("ERROR: pydot module not found.\nPlease install the python-pydot package.\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	else:
		msg = _("Unknown operating system.\n")
		sys.stdout.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)

#for plotting
try:
	import pylab
except ImportError, info:
	platform_sys = os.name
	if platform_sys in ('nt', 'mac'):
		msg = _("ERROR: Matplotlib module not found.\nhttp://sourceforge.net/projects/matplotlib/files/\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	elif platform_sys == 'posix':
		msg = _("ERROR: Matplotlib module not found.\nPlease install the python-matplotlib package.\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	else:
		msg = _("Unknown operating system.\n")
		sys.stdout.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)

# for graph
try:
	import networkx as nx
except ImportError, info:
	platform_sys = os.name
	if platform_sys in ('nt', 'mac'):
		msg = _("ERROR: Networkx module not found.\nhttp://networkx.lanl.gov/download/networkx/\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	elif platform_sys == 'posix':
		msg = _("ERROR: Networkx module not found.\nPlease install the python-networkx package.\n")
		sys.stderr.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)
	else:
		msg = _("Unknown operating system.\n")
		sys.stdout.write(msg)
		raise ImportError, "%s\n%s"%(msg,info)

### try to import psutil module if is installed.
try:
	from psutil import cpu_times
	def time(type="user"):
		#User CPU time is time spent on the processor running your program's code (or code in libraries);
		#system CPU time is the time spent running code in the operating system kernel on behalf of your program.
		# print cpu_times(True) for list of cpus

		return getattr(cpu_times(),type)

except ImportError:
	sys.stdout.write('Install psutil module for better cpu result of the plugin.\n The time module has been imported pending.')
	from time import time

### try if radon module is installed
try:
	from radon.raw import analyze
	import radon.metrics

	RAW_METRIC = True
	HALSTEAD_METRICS = True

except ImportError, info:
	platform_sys = os.name
	if platform_sys in ('nt', 'mac'):
		msg = _('ERROR: radon module not found.\nhttp://radon.readthedocs.org/en/latest/\n')
		sys.stderr.write(msg)
		#raise ImportError, "%s\n%s"%(msg,info)
	elif platform_sys == 'posix':
		msg = _('ERROR: radon module not found.\nPlease install the radon package.\n')
		sys.stderr.write(msg)
		#raise ImportError, "%s\n%s"%(msg,info)
	else:
		msg = _('Unknown operating system.\n')
		sys.stdout.write(msg)
		#raise ImportError, "%s\n%s"%(msg,info)

	RAW_METRIC = False
	HALSTEAD_METRICS = False

import pluginmanager
from Container import Block, CodeBlock, ContainerBlock, Diagram
from PlotGUI import PlotManager
from Domain.Basic.Object import Message
from Decorators import ProgressNotification
import ReloadModule

class Timer(object):
	def __init__(self, verbose=False, type='user'):
		'''	type must be 'user' or 'system' (see psutil above)
		'''
		self.verbose = verbose
		self._type = type

	def __enter__(self):
		self.start = time(self._type)
		return self

	def __exit__(self, *args):
		self.end = time(self._type)
		self.secs = self.end - self.start
		self.msecs = self.secs * 1000  # millisecs
		if self.verbose:
			print 'elapsed time: %f ms'%self.msecs

WRITE_DOT_TMP_FILE = True
WRITE_DYNAMIC_METRICS = False

def log(func):
	def wrapped(*args, **kwargs):

		try:
			#print "Entering: [%s] with parameters %s" % (func.__name__, args)
			try:

				func_name = func.__name__
				devs = func.im_self

				### function changes the state ?
				status_old = devs.state.get('status', None) if hasattr(devs, 'state') else None
				### output function send message using poke function ?
				send_output_old = map(lambda a: a.value, devs.myOutput.values())

				### on Window, psutil gives negative value...
				#ts = time()
				with Timer(type='user') as t:
					r =  func(*args, **kwargs)
				t_cpu = t.secs

				#te = time()
				#t_cpu = te-ts

				### function changes the state ?
				change_state = status_old and status_old != devs.state['status']
				### output function send message using poke function ?
				send_output = map(lambda a: a.value, devs.myOutput.values()) != send_output_old

				if not hasattr(devs,'texec'):
					setattr(devs,'texec', {func_name:[(0.0, t_cpu, change_state, send_output)]})
				else:

					if func_name in devs.texec.keys():
						### for number in axis
						try:
							ts = devs.timeLast + devs.elapsed
						### PyPDEVS has a tuple for timeLast
						except TypeError:
							ts = devs.timeLast[0] + devs.elapsed

						devs.texec[func_name].append((ts, t_cpu, change_state, send_output))
					else:
						devs.texec[func_name] = [(0.0, t_cpu, change_state, send_output)]

				#print devs, devs.texec,dict(map(lambda k,v: (k,v),devs.texec.keys(), map(lambda a: len(a), devs.texec.values())))
				return r
			except Exception, e:
				sys.stdout.write(_('Exception for Activity-Tracking plug-in in %s : %s' % (func.__name__, e)))
		finally:
			pass
			#print "Exiting: [%s]" % func.__name__
	return wrapped

def activity_tracking_decorator(inst):
	''' Decorator for the track of the activity of all atomic model transition function.
	'''

	for name, m in inspect.getmembers(inst, inspect.ismethod):
		if name in inst.getBlockModel().activity.values():
			setattr(inst, name, log(m))
			#setattr(inst,name,profile(m))
	return inst

def GetRawMetrics(m):
	""" Return Raw metrics values
	"""

	if RAW_METRIC:

		### class obj
		cls = m.__class__
		block = m.getBlockModel()

		### beware to use tab for devs code of models

		L = [eval("cls.%s"%fct) for fct in block.activity.values()]

		source_list = map(inspect.getsource, L)

		raw = {'loc':0, 'lloc':0, 'sloc':0, 'comments':0, 'multi':0, 'blank':0}
		for text in map(textwrap.dedent, source_list):
			a = analyze(text.strip())
			for key in raw:
				raw[key] += getattr(a, key)

		return raw

def GetHalsteadMetrics(m):
	""" Return Halstead values
	"""

	if HALSTEAD_METRICS:

		### class and graphic bloc obj
		cls = m.__class__
		block = m.getBlockModel()

		### beware to use tab for devs code of models

		L = [eval("cls.%s"%fct) for fct in block.activity.values()]

		source_list = map(inspect.getsource, L)

		#halstead = {'h1':0.0, 'h2':0.0, 'N1':0.0, 'N2':0.0, 'vocabulary':0.0, 'length':0.0, 'calculated_length':0.0, 'volume':0.0, 'difficulty':0.0, 'effort':0.0, 'time':0.0, 'bugs':0.0}
		halstead = {'vocabulary':0.0, 'length':0.0, 'calculated_length':0.0, 'volume':0.0}
		for text in map(textwrap.dedent, source_list):
			a = radon.metrics.h_visit(text.strip())
			for key in halstead:
				halstead[key] += getattr(a, key)

		return halstead

def delta_ext(devs):
	"""
	"""
	devs.extTransition()

def delta_int(devs):
	"""
	"""
	devs.intTransition()

def GetInitDEVSInstance(m):
	""" Get initial DEVS model instance
	"""

	cls = m.__class__
	block = m.getBlockModel()

	### devs model is instanciated dynamically from its cls string with default args (kwargs)
	kwargs = block.args
	devs = cls(**kwargs)

	devs.blockModel = block

	### add devs ports from block
	for i in xrange(block.input):
		devs.addInPort('in_%d'%i)

	for i in xrange(block.output):
		devs.addOutPort('out_%d'%i)

	### add simulation attributes
	devs.timeNext = devs.timeLast = devs.elapsed = 0.0

	return devs

def GetTimeIt(m):
	"""
	"""

	block = m.getBlockModel()

	### TODO : make it generic !
	### transition functions of the model below depend on the inputs !
	#######################################################
	### for WSUM model
	#devs.K = [1]*block.input
	#from numpy import zeros

	#devs.Xs = zeros(block.input)
	#devs.Mxs = zeros(block.input)
	#devs.Pxs = zeros(block.input)

	### for NLFucntion
	#devs.n = block.input
	#devs.u	= zeros(block.input)
	#devs.mu = zeros(block.input)
	#devs.pu = zeros(block.input)
	########################################################

	### dictionary with values to return (worst and best case)
	L = [0.0]*2

	### only of int transition is requested (some atomic model call the ext transition from int transition for initilaization (integrator))
	if 'intTransition' in block.activity.values():

		### Get DEVS instance from model by instancing the class
		devs = GetInitDEVSInstance(m)

		### add messages to inputs ports
		for p in devs.IPorts:
			devs.myInput[p] = None

		### define delta_int function from method
		f = functools.partial(delta_int, devs)

		tmp = timeit.repeat(f, number=5, repeat=100)

		########################
		### 	Best case	####
		########################

		### atomic model is activated one time
		### with one message on its first port

		### add message to the first input port
#		devs.myInput[devs.IPorts[0]] = Message([0,0,0], 0.0)

		### best case
		L[1] += min(tmp)*0.001

		########################
		### 	Wort case	####
		########################

		### atomic model is activated 1000
		### times with all messages on its ports

		### add messages to inputs ports
#		for p in devs.IPorts:
#			devs.myInput[p] = Message([0,0,0], 0.0)

		### delta_int called 1000 times (worst case)
		L[0] += max(tmp)*0.001

	### only of ext transition is requested
	if 'extTransition' in block.activity.values():

		### Get DEVS instance from model by instancing the class
		devs = GetInitDEVSInstance(m)

		### add messages to inputs ports
		for p in devs.IPorts:
			devs.myInput[p] = None

		########################
		### 	Best case	####
		########################

		### atomic model is activated one time
		### with one message on its first port

		### add message to the first input port
		devs.myInput[devs.IPorts[0]] = Message([random.random(),random.random(),random.random()], 0.0)

		### define delta_ext function from method
		f = functools.partial(delta_ext, devs)

		### best case
		L[1] += min(timeit.repeat(f, number=5, repeat=10000))*0.00001

		########################
		### 	Worst case	####
		########################

		### atomic model is activated 1000
		### times with all messages on its ports

		### Get DEVS and block instance from model by instancing the class
		devs = GetInitDEVSInstance(m)

		### add messages to inputs ports
		for p in devs.IPorts:
			devs.myInput[p] = Message([random.random(),random.random(),random.random()], 0.0)

		### define delta_ext function from method
		f = functools.partial(delta_ext, devs)

		### delta_int called 1000 times (wort case)
		L[0] += max(timeit.repeat(f, number=5, repeat=10000))*0.00001

	### add some message
	return L

def GetMacCabeMetric(m):
		"""
		"""

		complexity = 0.0

		try:
			import codepaths
		except ImportError, info:
			msg = _('ERROR: codepaths module not found.\n')
			sys.stderr.write(msg)
			return complexity
		else:

			### Get class of model
			cls = m.__class__
			block = m.getBlockModel()

			### mcCabe complexity
			### beware to use tab for devs code of models

			L = [eval("cls.%s"%fct) for fct in block.activity.values()]

			source_list = map(inspect.getsource, L)

			L_args = []

			for text in source_list:
				### textwrap for deleting the indentation

				ast = codepaths.compiler.parse(textwrap.dedent(text).strip())
				visitor = codepaths.PathGraphingAstVisitor()
				visitor.preorder(ast, visitor)

				for graph in visitor.graphs.values():
					complexity += graph.complexity()

					### write dot file
					if WRITE_DOT_TMP_FILE:
						worker(m.getBlockModel().label, str(m.myID), fct, str(graph.to_dot()))

			return complexity

def worker(label, ID, fct, txt):
	dot_path = os.path.join(tempfile.gettempdir(), "%s(%s)_%s.dot"%(label,str(ID),fct))

	#msg = "Starting write %s" % dot_path

	### write file in temp directory
	with open(dot_path,'wb') as f:
		f.write("graph {\n%s}"%txt)

######################################################################
###				Class Definition
######################################################################

class ActivityData:
	""" Class providing static and dynamic activity data.
	"""
	def __init__(self, model_list):
		''' Constructor
		'''
		### list of model selected in activity configuration panel
		self._model_list = model_list

		self._dynamic_data = None
		self._static_data = None

		### Porgress dialogue for long task
		self.progress_dlg = wx.ProgressDialog(_('Activity Reporter'),
								_("Loading activity data..."), parent=None,
								style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME)
		self.progress_dlg.Pulse()

	def EndProgressBar(self):
		### end of progress dialogue
		self.progress_dlg.Destroy()

	def setModelList(self, ml=[]):
		self._model_list = ml

	def getModelList(self):
		return self._model_list

	def getStaticData(self):
		""" Get static Activity data before simulation
		"""

		if not self._static_data:

			self.progress_dlg.UpdatePulse('Performing static activity data...')

			### static informations
			colLabels = [_('Model'), _('Id'), _('MCC'), _('length'),
						_('vocabulary'), _('volume'),
						_('calculated_length'), _('loc'),
						_('lloc'), _('sloc'),
						_('Worst_case[s]'), _('Best_case[s]')]

			model_list = self.getModelList()

			### Get all data
			model_name_list, model_id_list = zip(*map(lambda m : (m.getBlockModel().label, m.myID), model_list))

			self.progress_dlg.UpdatePulse(_('Static activity\n Performing MacCabe metric...'))
			mcCabe_list = map(GetMacCabeMetric, model_list)

			self.progress_dlg.UpdatePulse(_('Static activity\n Performing Halstead metric...'))
			halstead_list = map(GetHalsteadMetrics, model_list)
			h = map(lambda a: a['vocabulary'], halstead_list)
			N = map(lambda a: a['length'], halstead_list)
			vol = map(lambda a: a['volume'], halstead_list)
			calculated_lenght = map(lambda a: a['calculated_length'], halstead_list)

			self.progress_dlg.UpdatePulse(_('Static activity\n Performing Raw metric...'))
			raw_list = map(GetRawMetrics, model_list)
			loc = map(lambda a: a['loc'], raw_list)
			lloc = map(lambda a: a['lloc'], raw_list)
			sloc = map(lambda a: a['sloc'], raw_list)

			self.progress_dlg.UpdatePulse(_('Static activity\n Performing estimated execution time...'))
			timeIt_list = map(GetTimeIt, model_list)

			self.progress_dlg.UpdatePulse(_('Formating activity data...'))
			### data used to initialize table
			data = map(lambda a,b,c,d,e,f,g,h,i,j,k : [a,b,c,d,e,f,g,h,i,j,k[0],k[1]], model_name_list, model_id_list, mcCabe_list, h, N, vol, calculated_lenght, loc, lloc, sloc, timeIt_list)

			### row label is the name of selected model
			rowLabels = map(lambda a: str(a), range(len(map(lambda b: b[0], data))))

			self._static_data = (data, rowLabels, colLabels)

		return self._static_data

	def getDynamicData(self, H=0.0):
		""" Get Dynamic activity data after simulation
		"""

		self.progress_dlg.UpdatePulse(_('Performing dynamic data...'))

		if not self._dynamic_data:

			### dynamic informations
			colLabels = [_('Model'), _('QAint'), _('CSint'), _('WFIint') ,_('QAext'), _('CSext'), _('WFIext'), _('QAout'), _('QPoke'), _('QAta'), _('Activity'), _('QActivity'), _('WActivity'), _('CPU[s]')]

			quantitative_activity_list = []
			cpu_activity_list = []
			weighted_activity_list = []
			model_list = []
			quantitative_activity_int_list = []
			quantitative_activity_ext_list = []
			quantitative_activity_out_list = []
			quantitative_activity_ta_list = []
			state_change_int_list = []
			state_change_ext_list = []
			send_output_list = []
			wait_for_imminent_int_list = []
			wait_for_imminent_ext_list = []


			for m in self.getModelList():

				self.progress_dlg.UpdatePulse(_('Dynamic activity\n Performing activity mesures for %s...')%m.getBlockModel().label)

				quantitative_activity = 0
				quantitative_activity_int = 0
				quantitative_activity_ext = 0
				quantitative_activity_out = 0
				quantitative_activity_ta = 0
				state_change_int = 0
				state_change_ext = 0
				send_output = 0
				cpu_activity = 0.0
				weighted_activity = 0.0
				wait_for_imminent_int = 0.0
				qiat_for_imminent_ext = 0.0
				#texec_list = m.texec.values()

				for fct,d in m.texec.items():
					v = len(d)
					quantitative_activity+=v

					if fct == "intTransition":
						quantitative_activity_int = v
						state_change_int = len(filter(lambda c: c[2] == True, d))
						t = map(lambda a: a[0],d)
						print "int", [x[0]-x[1] for x in zip(t[1:],t[:-1])], t
						wait_for_imminent_int = sum([x[0]-x[1] for x in zip(t[1:],t[:-1])])
					elif fct == "extTransition":
						quantitative_activity_ext = v
						state_change_ext = len(filter(lambda c: c[2] == True, d))
						t = map(lambda a: a[0],d)
						print "ext", [x[0]-x[1] for x in zip(t[1:],t[:-1])], t
						wait_for_imminent_ext = sum([x[0]-x[1] for x in zip(t[1:],t[:-1])])
					elif fct == "outputFnc":
						quantitative_activity_out = v
						send_output = len(filter(lambda c: c[3] == True, d))
					else:
						quantitative_activity_ta = v

					cpu_activity+=sum(map(lambda c: c[1], d))
					### TODO round for b-a ???
					weighted_activity+=d[-1][0]-d[0][0]

				quantitative_activity_list.append(quantitative_activity)
				cpu_activity_list.append(cpu_activity)
				weighted_activity_list.append(weighted_activity)
				model_list.append(m.getBlockModel().label)
				quantitative_activity_int_list.append(quantitative_activity_int)
				quantitative_activity_ext_list.append(quantitative_activity_ext)
				quantitative_activity_out_list.append(quantitative_activity_out)
				quantitative_activity_ta_list.append(quantitative_activity_ta)
				state_change_int_list.append(state_change_int)
				state_change_ext_list.append(state_change_ext)
				send_output_list.append(send_output)
				wait_for_imminent_ext_list.append(wait_for_imminent_ext)
				wait_for_imminent_int_list.append(wait_for_imminent_int)

				self.setDataToDEVSModel(m, {'quantitative' : quantitative_activity, \
											'cpu' : cpu_activity, \
											'weighted' : weighted_activity
											})

			### if models have been simulated during a minimum time H
			if H > 0.0:
				### prepare data to populate grid
				data =  map(lambda label, qaint, csint, wfiint ,qaext, csext, wfiext, qaout, send, qata, qa, cpu, wa: (label, qaint, csint, wfiint, qaext, csext, wfiext, qaout, send, qata, qa, qa/H, wa/H, cpu/qa), \
							model_list,\
                            quantitative_activity_int_list, \
                            state_change_int_list, \
                            wait_for_imminent_int_list,\
                            quantitative_activity_int_list, \
                            state_change_ext_list, \
                            wait_for_imminent_ext_list,\
                            quantitative_activity_out_list, \
                            send_output_list, \
                            quantitative_activity_ta_list, \
                            quantitative_activity_list, \
							cpu_activity_list, \
							weighted_activity_list)
			else:
				data =  map(lambda label, qaint, csint, wfiint, qaext, csext, wfiext, qaout, send, qata, qa, cpu, wa: (label, qaint, csint, qaext, csext, qaout, send, qata, qa, 0.0, 0.0, 0.0), \
                            model_list,\
                            quantitative_activity_int_list, \
                            state_change_int_list, \
                            quantitative_activity_int_list, \
                            quantitative_activity_ext_list, \
                            state_change_ext_list, \
                            quantitative_activity_ext_list, \
                            quantitative_activity_out_list, \
                            send_output_list, \
                            quantitative_activity_ta_list, \
							quantitative_activity_list, \
							cpu_activity_list, \
							weighted_activity_list)

			### row label is the name of selected model
			rowLabels = map(lambda a: str(a), range(len(map(lambda b: b[0], data))))

			self._dynamic_data = (data, rowLabels, colLabels)

		return self._dynamic_data

	def setDataToDEVSModel(self, model, d={}):
		""" Embed all information about activity in a new attribute of model (named 'activity')
		"""

		if not hasattr(model, 'activity'):
			setattr(model, 'activity', d)
		else:
			model.activity.update(d)

class GenericTable(wx.grid.PyGridTableBase):
	def __init__(self, data, rowLabels=None, colLabels=None):
		wx.grid.PyGridTableBase.__init__(self)
		self.data = data
		self.rowLabels = rowLabels
		self.colLabels = colLabels

		# we need to store the row length and column length to
		# see if the table has changed size
		self._rows = self.GetNumberRows()
		self._cols = self.GetNumberCols()

	def GetNumberRows(self):
		return len(self.data)

	def GetNumberCols(self):
		return len(self.data[0])

	def GetColLabelValue(self, col):
		if self.colLabels:
			return self.colLabels[col]

	def GetRowLabelValue(self, row):
		if self.rowLabels:
			return self.rowLabels[row]

	def IsEmptyCell(self, row, col):
		return False

	def GetValue(self, row, col):
		return self.data[row][col]

	def SetValue(self, row, col, value):
		pass

	def UpdateValues(self, grid):
		"""Update all displayed values"""
		# This sends an event to the grid table to update all of the values
		msg = wx.grid.GridTableMessage(self, wx.grid.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
		grid.ProcessTableMessage(msg)

	def DeleteCols(self, cols):
		"""
		cols -> delete the columns from the dataset
		cols hold the column indices
		"""
		# we'll cheat here and just remove the name from the
		# list of column names.  The data will remain but
		# it won't be shown
		deleteCount = 0
		cols = cols[:]
		cols.sort()

		for i in cols:
			self.colLabels.pop(i-deleteCount)
			# we need to advance the delete count
			# to make sure we delete the right columns
			deleteCount += 1

		if not len(self.colLabels):
			self.data = []

	def DeleteRows(self, rows):
		"""
		rows -> delete the rows from the dataset
		rows hold the row indices
		"""
		deleteCount = 0
		rows = rows[:]
		rows.sort()

		for i in rows:
			self.data.pop(i-deleteCount)
			# we need to advance the delete count
			# to make sure we delete the right rows
			deleteCount += 1

	def SortColumn(self, col):
		"""
		col -> sort the data based on the column indexed by col
        """
		name = self.colLabels[col]
		_data = []

		for row in self.data:
			rowname, entry = row
			_data.append((entry.get(name, None), row))
			_data.sort()
			self.data = []

			for sortvalue, row in _data:
				self.data.append(row)

class ActivityReport(wx.Frame):

	def __init__(self, parent, id, size, title='', style = wx.DEFAULT_FRAME_STYLE, master=None):
		# begin wxGlade: ActivityReport.__init__
		wx.Frame.__init__(self, parent, id, title=title, name=_('Tracking'), style=style)

		self._title = title
		self._master = master
		self.parent = parent

		self.panel = wx.Panel(self, wx.ID_ANY)

		self.ReportGrid = wx.grid.Grid(self.panel, wx.ID_ANY, size=(1, 1))

		self.timer = wx.Timer(self)

		self.__set_properties()
		self.__do_layout()

		self.timer.Start(2000, oneShot=wx.TIMER_CONTINUOUS)

		self.Bind(wx.EVT_TIMER, self.OnUpdate)
		self.Bind(wx.grid.EVT_GRID_CELL_LEFT_DCLICK,self.OnDClick, id=self.ReportGrid.GetId())
		self.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK,self.OnRightClick, id=self.ReportGrid.GetId())
		self.ReportGrid.GetGridColLabelWindow().Bind(wx.EVT_MOTION, self.onMouseOverColLabel)
		self.Bind(wx.EVT_BUTTON, self.OnRefresh, id=self.btn.GetId())
		self.Bind(wx.EVT_TOGGLEBUTTON, self.OnDynamicRefresh, id=self.tbtn.GetId())
		self.Bind(wx.EVT_BUTTON, self.OnExport, id=self.ebtn.GetId())

		#self.Bind(wx.grid.EVT_GRID_LABEL_RIGHT_CLICK, self.OnLabelRightClicked)

		# end wxGlade

	def __set_properties(self):
		# begin wxGlade: ActivityReport.__set_properties

		self.SetTitle(self._title)

		data = []

		### get user configuration ofr static or dynamic execution
		diagram = self.parent.master.getBlockModel()

		if hasattr(diagram, 'activity_flags'):
			self.static_user_flag, self.dynamic_user_flag = diagram.activity_flags
		else:
			self.static_user_flag = self.dynamic_user_flag = True

		### after simulation, we can display the static and dynamic activity informations
		if self._master and self.dynamic_user_flag:

			L = GetFlatDEVSList(self._master, [])

			### if model have texec attribute, then its selected in the properties panel
			model_list = filter(lambda a: hasattr(a, 'texec'), L)

			### if at least one model has been selected in Activity configuration dialogue
			if len(model_list) != 0:

				d = ActivityData(model_list)

				### A=(Aint+Aext)/H
				H=self._master.timeLast if self._master.timeLast <= self._master.FINAL_TIME else self._master.FINAL_TIME

				Ddata, DrowLabels, DcolLabels = d.getDynamicData(H)

				if self.static_user_flag:
					Sdata, SrowLabels, ScolLabels = d.getStaticData()

					### update data ([1:] to delete the label of the model in case of dynamic data with static data)
					ScolLabels.extend(DcolLabels[1:])

					for i in range(len(Sdata)):
						### ([1:] to delete the label of the model in case of dynamic data with static data)
						Sdata[i].extend(Ddata[i][1:])

					###
					data = Sdata
					colLabels = ScolLabels
				else:
					data = Ddata
					colLabels = DcolLabels

				rowLabels = map(lambda a: str(a), range(len(map(lambda b: b[0], data))))

				d.EndProgressBar()

			### no models are selected from activity plug-in properties panel
			else:
				self.ReportGrid.Show(False)
				wx.MessageBox(_('Please select at least one model in activity configuration panel.'), _('Info'), wx.OK|wx.ICON_INFORMATION)

		### static data
		elif self.static_user_flag:

			### get DEVS model form diagram
			diagram.Clean()
			self._master = Diagram.makeDEVSInstance(diagram)

			### for static infrmations, all model are considered
			L = GetFlatDEVSList(self._master, [])

			model_list = filter(lambda a: hasattr(a.getBlockModel(), 'activity'), L)

			if len(model_list) != 0:
				d = ActivityData(model_list)
				data, rowLabels, colLabels = d.getStaticData()
				d.EndProgressBar()
			else:
				self.ReportGrid.Show(False)
				wx.MessageBox(_('Please select at least one model in activity configuration panel.'), _('Info'), wx.OK|wx.ICON_INFORMATION)

		### populate the grid from data
		if len(data) != 0:

			tableBase = GenericTable(data, rowLabels, colLabels)

			self.ReportGrid.CreateGrid(10, len(colLabels))
			for i in range(len(colLabels)):
				self.ReportGrid.SetColLabelValue(i, colLabels[i])

			self.ReportGrid.SetTable(tableBase)
			self.ReportGrid.EnableEditing(0)
			self.ReportGrid.SetColLabelSize(60)
			self.ReportGrid.SetLabelTextColour('#0000ff') # Blue
			self.ReportGrid.AutoSize()

	###
	def __do_layout(self):
		"""
		"""

		sizer_1 = wx.BoxSizer(wx.VERTICAL)
		sizer_2 = wx.BoxSizer(wx.HORIZONTAL)

		self.tbtn = wx.ToggleButton(self.panel,  wx.NewId(), _('Auto-Refresh'))
		self.tbtn.SetValue(True)

		self.btn = wx.Button(self.panel, wx.NewId(), _('Refresh'))
		self.btn.Enable(False)

		self.ebtn = wx.Button(self.panel, wx.NewId(), _('Export'))

		sizer_2.Add(self.tbtn, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 3, 3)
		sizer_2.Add(self.btn, 0, wx.RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 3, 3)
		sizer_2.Add(self.ebtn, 0, wx.RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 3, 3)

		sizer_1.Add(self.ReportGrid, 1, wx.EXPAND|wx.ALL, 5)
		sizer_1.Add(sizer_2, 0, wx.BOTTOM|wx.EXPAND|wx.ALL, 5)

#		sizer = wx.BoxSizer(wx.VERTICAL)

#		self.panel.SetSizer(sizer_1)
#		self.panel.Layout()
#		sizer_1.Fit(self.panel)
#		sizer.Add( self.panel, 1, wx.EXPAND |wx.ALL, 0 )

		#self.ReportGrid.Bind(wx.EVT_SIZE, self.OnSize)
		#self.SetSizer(sizer)
		#self.SetSize(self.ReportGrid.GetBestSize()+(10,10))
		#self.Layout()

		#####################################
		self.panel.SetSizer(sizer_1)
		self.Layout()

		### http://wxpython-users.1045709.n5.nabble.com/Auto-size-panel-to-fit-grid-td2344890.html

		#self.panel.Fit()
		self.SetClientSize(self.ReportGrid.GetBestSize()+(20,100))


	def OnSize(self, event):
		width, height = self.GetClientSizeTuple()
		c=self.ReportGrid.GetNumberCols()
		for col in range(c):
			self.ReportGrid.SetColSize(col, width/(c + 1))

	def OnLabelRightClicked(self, evt):
		# Did we click on a row or a column?
		row, col = evt.GetRow(), evt.GetCol()
		if row == -1: self.colPopup(col, evt)
		elif col == -1: self.rowPopup(row, evt)

	def colPopup(self, col, evt):
		"""(col, evt) -> display a popup menu when a column label is
			right clicked
		"""
		x = self.ReportGrid.GetColSize(col)/2
		menu = wx.Menu()
		id1 = wx.NewId()
		sortID = wx.NewId()

		xo, yo = evt.GetPosition()
		self.ReportGrid.SelectCol(col)
		cols = self.ReportGrid.GetSelectedCols()
		self.ReportGrid.Refresh()
		menu.Append(id1, 'Delete Col(s)')
		menu.Append(sortID, 'Sort Column')

		def delete(event, self=self, col=col):
			cols = self.ReportGrid.GetSelectedCols()
			self.ReportGrid.GetTable().DeleteCols(cols)
			self.ReportGrid.Reset()

		def sort(event, self=self, col=col):
			self.ReportGrid.GetTable().SortColumn(col)
			self.ReportGrid.Reset()

		self.ReportGrid.Bind(wx.EVT_MENU, delete, id=id1)

		if len(cols) == 1:
			self.ReportGrid.Bind(wx.EVT_MENU, sort, id=sortID)

		self.ReportGrid.PopupMenu(menu)
		menu.Destroy()
		return

	def OnExport(self, event):
		"""	csv file exporting
		"""

		wcd = _('CSV files (*.csv)|*.csv|Text files (*.txt)|*.txt|All files (*.*)|*.*')
		home = os.getenv('USERPROFILE') or os.getenv('HOME') or HOME_PATH
		export_dlg = wx.FileDialog(self, message=_('Choose a file'), defaultDir=home, defaultFile='data.csv', wildcard=wcd, style=wx.SAVE|wx.OVERWRITE_PROMPT)
		if export_dlg.ShowModal() == wx.ID_OK:
			fileName = export_dlg.GetPath()
			try:
				spamWriter = csv.writer(open(fileName, 'w'), delimiter=' ', quotechar='|', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)

				spamWriter.writerow([self.ReportGrid.GetColLabelValue(i).replace('\n', '') for i in xrange(self.ReportGrid.GetNumberCols())])

				for row in xrange(self.ReportGrid.GetNumberRows()):
					spamWriter.writerow([self.ReportGrid.GetCellValue(row, i) for i in xrange(self.ReportGrid.GetNumberCols())])

			except Exception, info:
				dlg = wx.MessageDialog(self, _('Error exporting data: %s\n'%info), _('Export Manager'), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()

			dial = wx.MessageDialog(self, _('Export completed'), _('Export Manager'), wx.OK|wx.ICON_INFORMATION)
			dial.ShowModal()
			export_dlg.Destroy()

	def OnDynamicRefresh(self, event):
		""" Check-box has been checked
		"""

		### update the button status
		self.btn.Enable(not self.tbtn.GetValue())

		### update the timer
		if not self.tbtn.GetValue():
			self.timer.Stop()
		else:
			self.timer.Start(2000, oneShot=wx.TIMER_CONTINUOUS)

	def OnRefresh(self, event):
		""" Button Refresh has been pushed
		"""
		if not self.timer.IsRunning():
			self.OnUpdate(event)

	###
	def OnUpdate(self, evt):
		"""
		"""

		sim_thread = self.parent.thread

		### update only of simulation thread is alive
		if sim_thread and sim_thread.isAlive():

			### needed from GetData
			self._master = self.parent.master

			### if model_list is not defined, set_properties method must be performed
			### in order to create the ReportGrid object
			if not hasattr(self, 'model_list'):
				self.__set_properties()

			data = self.GetData()
			table = self.ReportGrid.GetTable()

			### update table from data
			for i,c in enumerate(data):
				### change value of the cell
				table.SetValue(i, 2, c[0])
				table.SetValue(i, 3, c[1])
				table.SetValue(i, 4, c[2])

				try:
					### change the value in the data attribute of the table
					table.data[i][2] = c[0]
					table.data[i][3] = c[1]
					table.data[i][4] = c[2]
				except:
					pass

				if WRITE_DYNAMIC_METRICS:
					with open(os.path.join(tempfile.gettempdir(),str(table.GetValue(i,0))+'_CPU_.csv'), 'a') as file1:
						file1.write(str(c[2])+'\n')

					with open(os.path.join(tempfile.gettempdir(),str(table.GetValue(i,0))+'_QA_.csv'), 'a') as file2:
						file2.write(str(c[0])+'\n')

			self.ReportGrid.SetTable(table)
			self.ReportGrid.Refresh()

	def onMouseOverColLabel(self, event):
		"""
		Displays a tool-tip when mousing over certain column labels
		"""
		x = event.GetX()
		y = event.GetY()
		col = self.ReportGrid.XToCol(x, y)

		L = [_('Name of model')
			]

		if self.static_user_flag:

			L += [ _('Identity number'),
				_('MacCabe\'s Cyclomatic Complexity'),
				_('Program vocabulary: η=η1+η2'),
				_('Program length: N=N1+N2'),
				_('Volume: V=Nlog2η'),
				_('Calculated program length: Nˆ=η1log2η1+η2log2η2'),
				_('Number of lines of code (total)'),
				_('Number of logical lines of code'),
				_('Number of source lines of code (not necessarily corresponding to the LLOC)'),
				_('Worst estimated execution time from timeit module'),
				_('Best estimated execution time from timeit module')]

		if self.dynamic_user_flag:

			L+=[_('Number of internal transition function activation (Aint)'),
				_('Number of state change due to the internal transition function activation'),
				_('Sum of time spent waiting an internal message'),
				_('Number of external transition function activation (Aext)'),
				_('Number of state change due to the external transition function activation'),
				_('Sum of time spent waiting an external message'),
				_('Number of output function activation (Aout)'),
				_('Number of message (using poke function) sended by the output function'),
				_('Number of time advance function activation (Ata)'),
				_('Number of Activation'),
				_('Quantitative Activity (A=Aint+Aext+Aout+Ata) which depends on the total execution time'),
				_('Weighted Activity (Zeigler def)'),
				_('Time spent running your program\'s code')]

		try:
			msg=L[col]
		except:
			msg = ''

		self.ReportGrid.GetGridColLabelWindow().SetToolTipString(msg)

		event.Skip()

	def showPopupMenu(self, event):
		"""
		Create and display a pop-up menu on right-click event
		"""

		win  = event.GetEventObject()

		### make a menu
		self.popupmenu = wx.Menu()
		# Show how to put an icon in the menu
		#plot_item = wx.MenuItem(self.popupmenu, wx.NewId(), _("Plot"))
		#table_item = wx.MenuItem(self.popupmenu, wx.NewId(), _("Table"))

		graph_item = wx.MenuItem(self.popupmenu, wx.NewId(), _('Graph'))

		#self.popupmenu.AppendItem(plot_item)
		#self.popupmenu.AppendItem(table_item)

		### The dot file is created during the process that create the grid
		if not WRITE_DOT_TMP_FILE:
			graph_item.Enable(False)

		self.popupmenu.AppendItem(graph_item)

		#self.Bind(wx.EVT_MENU, self.OnPopupItemPlot, plot_item)
		#self.Bind(wx.EVT_MENU, self.OnPopupItemTable, table_item)

		self.Bind(wx.EVT_MENU, self.OnPopupItemGraph, graph_item)

		# Popup the menu.  If an item is selected then its handler
		# will be called before PopupMenu returns.
		win.PopupMenu(self.popupmenu)
		self.popupmenu.Destroy()

	def OnPopupItemPlot(self, event):
		"""
		"""
		#item = self.popupmenu.FindItemById(event.GetId())
        #text = item.GetText()
        pass

	def OnPopupItemGraph(self, event):

		for row in self.ReportGrid.GetSelectedRows():
			label = self.ReportGrid.GetCellValue(row,0)
			id = self.ReportGrid.GetCellValue(row,1)

			### plot the graph
			### TODO link with properties frame
			for fct in ('extTransition','intTransition', 'outputFnc', 'timeAdvance'):
				filename = "%s(%s)_%s.dot"%(label,str(id),fct)
				path = os.path.join(tempfile.gettempdir(), filename)

				### if path exist
				if os.path.exists(path):
					graph = pydot.graph_from_dot_file(path)
					filename_png = os.path.join(tempfile.gettempdir(),"%s(%s)_%s.png"%(label,str(id),fct))
					graph.write_png(filename_png, prog='dot')

					pylab.figure()
					img = pylab.imread(filename_png)
					pylab.imshow(img)

					fig = pylab.gcf()
					fig.canvas.set_window_title(filename)

					pylab.axis('off')
					pylab.show()

					### TODO make analysis to implement probability based on path length
					#nx.draw(g)
					#g = nx.Graph(nx.read_dot(path))
					#distance =nx.all_pairs_shortest_path_length(g)
					#print distance

	def OnPopupItemTable(self, event):
		"""
		"""
		#item = self.popupmenu.FindItemById(event.GetId())
        #text = item.GetText()
		pass

	def OnRightClick(self, evt):
		self.showPopupMenu(evt)

	def OnDClick(self, evt):

		row = evt.GetRow()
		col = evt.GetCol()

		### label of model has been clicked on colon 0 and we plot the quantitative activity
		main = wx.GetApp().GetTopWindow()
		nb2 = main.GetDiagramNotebook()
		currentPage = nb2.GetCurrentPage()
		diagram = currentPage.diagram
		Plot(diagram, self.ReportGrid.GetCellValue(row,0))

@pluginmanager.register('START_ACTIVITY_TRACKING')
def start_activity_tracking(*args, **kwargs):
	""" Start the definition of the activity attributes for all selected block model
	"""

	master = kwargs['master']
	parent = kwargs['parent']

	for devs in GetFlatDEVSList(master, []):
		block = devs.getBlockModel()
		if hasattr(block, 'activity'):
			devs = activity_tracking_decorator(devs)

@pluginmanager.register('VIEW_ACTIVITY_REPORT')
def view_activity_report(*args, **kwargs):
	""" Start the definition of the activity attributs for all selected block model
	"""

	master = kwargs['master']
	parent = kwargs['parent']

	frame = ActivityReport(parent, wx.ID_ANY, size=(1250, 400), title=_('Activity-Tracking Reporter'), master = master)
	frame.Show()

def GetFlatDEVSList(coupled_devs, l=[]):
	""" Get the flat list of devs model composing coupled_devs (recursively)
	"""

	from DomainInterface.DomainBehavior import DomainBehavior
	from DomainInterface.DomainStructure import DomainStructure

	for devs in coupled_devs.componentSet:
		if isinstance(devs, DomainBehavior):
			l.append(devs)
		elif isinstance(devs, DomainStructure):
			l.append(devs)
			GetFlatDEVSList(devs,l)
	return l

def GetFlatShapesList(diagram,L):
	""" Get the list of shapes recursively
	"""
	for m in diagram.GetShapeList():
		if isinstance(m, CodeBlock):
			L.append(m.label)
		elif isinstance(m, ContainerBlock):
			 GetFlatShapesList(m,L)
	return L

def Plot(diagram, selected_label):

	master = diagram.getDEVSModel()

	if master is not None:

		### for all devs models with texec attribute (activity tracking has been activated for these type of models)
		for m in GetFlatDEVSList(master, []):
			label = m.getBlockModel().label
			### mdoel is schecked and selected
			if hasattr(m, 'texec') and selected_label == label:
				### add the results attribute specific for quickscope familly models
				setattr(m,'results', m.texec)
				### no fusion because we need to have separate window (if True we have one window)
				setattr(m,'fusion',False)
				### to have getBlockModel attribut, the codeBlock graphical model is introduced
				cb = CodeBlock(label)
				cb.setDEVSModel(m)

				### get canvas from main window
				main = wx.GetApp().GetTopWindow()
				canvas = main.nb2.GetCurrentPage()
				### plot frame has been invoked with a manager (dynamic or static plotting)
				PlotManager(canvas, _('CPU Activity'), m, xl = 'Time [s]', yl = 'CPU time')

	else:
		dial = wx.MessageDialog(event.GetEventObject(), _('Master DEVS Model is None!\nGo to the simulation process in order to perform activity tracking.'), _('Plot Manager'), wx.OK | wx.ICON_EXCLAMATION)
		dial.ShowModal()

def Config(parent):
	""" Plugin settings frame.
	"""

	global cb1
	global cb2
	global cb3
	global diagram

	main = wx.GetApp().GetTopWindow()
	nb2 = main.GetDiagramNotebook()
	currentPage = nb2.GetCurrentPage()
	diagram = currentPage.diagram
	master = None

	frame = wx.Frame(parent, wx.ID_ANY, title = _('Activity Tracking'), style = wx.DEFAULT_FRAME_STYLE | wx.CLIP_CHILDREN | wx.STAY_ON_TOP)
	panel = wx.Panel(frame, wx.ID_ANY)

	#lst_1  = map(lambda a: a.label, filter(lambda s: isinstance(s, CodeBlock), diagram.GetShapeList()))
	lst_1 = GetFlatShapesList(diagram, [])
	lst_2  = ('timeAdvance', 'outputFnc', 'extTransition', 'intTransition')
	lst_3  = ('static', 'dynamic')

	vbox = wx.BoxSizer(wx.VERTICAL)
	hbox = wx.BoxSizer(wx.HORIZONTAL)
	hbox2 = wx.BoxSizer(wx.HORIZONTAL)

	st = wx.StaticText(panel, wx.ID_ANY, _('Select models and functions to track:'), (10,10))

	cb1 = wx.CheckListBox(panel, wx.ID_ANY, (10, 30), wx.DefaultSize, lst_1, style=wx.LB_SORT)
	cb2 = wx.CheckListBox(panel, wx.ID_ANY, (10, 30), wx.DefaultSize, lst_2)
	cb3 = wx.CheckListBox(panel, wx.ID_ANY, (10, 30), wx.DefaultSize, lst_3)

	selBtn = wx.Button(panel, wx.ID_SELECTALL)
	desBtn = wx.Button(panel, wx.ID_ANY, _('Deselect All'))
	okBtn = wx.Button(panel, wx.ID_OK)
	#reportBtn = wx.Button(panel, wx.ID_ANY, _('Report'))

	hbox2.Add(cb1, 1, wx.EXPAND, 10, 10)
	hbox2.Add(cb2, 1, wx.EXPAND, 10, 10)
	hbox2.Add(cb3, 1, wx.EXPAND, 10, 10)

	hbox.Add(selBtn, 0, wx.LEFT, 5, 10)
	hbox.Add(desBtn, 0, wx.CENTER, 5, 10)
	hbox.Add(okBtn, 0, wx.RIGHT, 5, 10)

	vbox.Add(st, 0, wx.ALL, 5)
	vbox.Add(hbox2, 1, wx.EXPAND, 10, 10)
	vbox.Add(hbox, 0, wx.CENTER, 10, 10)

	### si des modèles sont deja activés pour le plugin il faut les checker
	num = cb1.GetCount()
	L1 = [] ### liste des shapes à checker
	L2 = {} ### la liste des function tracer (identique pour tous les block pour l'instant)
	for index in range(num):
		block = diagram.GetShapeByLabel(cb1.GetString(index))
		if hasattr(block, 'activity'):
			L1.append(index)
			L2[block.label] = block.activity.keys()

	if L1 != []:
		cb1.SetChecked(L1)
		### tout les block on la meme liste de function active pour le trace, donc on prend la première
		cb2.SetChecked(L2.values()[0])

	### ckeck par defaut delta_ext et delta_int
	if L2 == {}:
		cb2.SetChecked([2,3])

	if hasattr(diagram, 'activity_flags'):
		cb3.SetChecked([i for i,v in enumerate(diagram.activity_flags) if v])
	else:
		### all checked
		cb3.SetChecked(range(len(lst_3)))

	panel.SetSizer(vbox)
	vbox.Fit(frame)

	def OnPlot(event):
		''' Bar Plot for the activity tracking performed
		'''
		Plot(diagram, cb1.GetString(cb1.GetSelection()))

	def OnSelectAll(evt):
		""" Select All button has been pressed and all plugins are enabled.
		"""
		cb1.SetChecked(range(cb1.GetCount()))

	def OnDeselectAll(evt):
		""" Deselect All button has been pressed and all plugins are disabled.
		"""
		cb1.SetChecked([])

	def OnOk(evt):
		btn = evt.GetEventObject()
		frame = btn.GetTopLevelParent()
		num1 = cb1.GetCount()
		num2 = cb2.GetCount()

		for index in range(num1):
			label = cb1.GetString(index)

			shape = diagram.GetShapeByLabel(label)
			activity_condition = hasattr(shape,'activity')

			assert(isinstance(shape, Block))

			if cb1.IsChecked(index):
				### dictionnaire avec des clees correspondant aux index de la liste de function de transition et avec des valeurs correspondant aux noms de ces fonctions
				D = dict([(index, cb2.GetString(index)) for index in range(num2) if cb2.IsChecked(index)])
				if not activity_condition :
					setattr(shape, 'activity', D)
				else:
					shape.activity = D
			elif activity_condition:
				del shape.activity

		### add static or dynamic cb info into the activity attribute of the diagram
		v = [cb3.IsChecked(0), cb3.IsChecked(1)]
		if not hasattr(diagram, 'activity_flags'):
			setattr(diagram, 'activity_flags', v)
		else:
			diagram.activity_flags = v

		frame.Destroy()

	selBtn.Bind(wx.EVT_BUTTON, OnSelectAll)
	desBtn.Bind(wx.EVT_BUTTON, OnDeselectAll)
	okBtn.Bind(wx.EVT_BUTTON, OnOk)

	def showPopupMenu(event):
		"""
		Create and display a popup menu on right-click event
		"""

		win  = event.GetEventObject()

		### make a menu
		menu = wx.Menu()
		# Show how to put an icon in the menu
		item = wx.MenuItem(menu, wx.NewId(), 'Aext')
		menu.AppendItem(item)
		menu.Append(wx.NewId(), 'Aint')
		menu.Append(wx.NewId(), 'A=Aext+Aint')

		# Popup the menu.  If an item is selected then its handler
		# will be called before PopupMenu returns.
		win.PopupMenu(menu)
		menu.Destroy()

	def OnRightClickCb1(evt):
		showPopupMenu(evt)

	def OnRightDClickCb1(evt):
		OnPlot(evt)

	### 1. Register source's EVT_s to inOvoke launcher.
	cb1.Bind(wx.EVT_RIGHT_DOWN, OnRightClickCb1)
	cb1.Bind(wx.EVT_LEFT_DCLICK, OnRightDClickCb1)

	frame.CenterOnParent(wx.BOTH)
	frame.Show()

def UnConfig():
	""" Reset the plugin effects on the TransformationADEVS model
	"""

	global cb1
	global cb2
	global cb3
	global diagram

	main = wx.GetApp().GetTopWindow()
	nb2 = main.GetDiagramNotebook()
	currentPage = nb2.GetCurrentPage()
	diagram = currentPage.diagram

	lst  = map(lambda a: a.label, filter(lambda s: isinstance(s, CodeBlock), diagram.GetShapeList()))

	for label in lst:
		shape = diagram.GetShapeByLabel(label)
		if hasattr(shape, 'activity'):
			del shape.activity
