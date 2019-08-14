import numpy as np
import scipy
class network:
	def __init__(self,inp_list):
		self.we=[]
		self.inp=inp_list
		###weight
		for i in range(len(inp_list)-1):
			w=np.random.normal(0.0,pow(inp_list[i+1],-0.5),(inp_list[i+1],inp_list[i]))
			
			self.we.append(w)
		#####biases
		self.biases=[]
		for j in range(len(inp_list)-1):
			b=np.random.normal(0.0,pow(inp_list[j+1],-0.5),(inp_list[j+1],1))
			self.biases.append(b)
		###activation function###
		self.lr=1.5
		
	def sigmoid(self,z):
		return (1/(1+np.exp(-z)))
	
	#######query part#######
	def query(self,input):
		fin_list=[]
		ip_val=np.array(input)
		val=np.reshape(ip_val,(self.inp[0],1))
		for i in range(len(self.we)):
			fin_list.append(val)
			z=np.dot(self.we[i],val)+self.biases[i]
			a=self.sigmoid(z)
			val=a
		fin_list.append(val)
		return fin_list[-1]
		
	def dsigmoid(self,x):
		return (x*(1-x))
		
	#######train########
	def train(self,input,target,it):
		fin_list=[]
		ip=np.array(input)
		op=np.array(target)
		val=np.reshape(ip,(self.inp[0],1))
		opt=np.reshape(op,(self.inp[-1],1))
		for i in range(len(self.inp)-1):
			fin_list.append(val)
			print(self.we[i].shape,val.shape)
			z=np.dot(self.we[i],val)+self.biases[i]
			a=self.sigmoid(z)
			val=a
		fin_list.append(val)
		out=fin_list[-1]
		err=target-out
		if it%100==0:
			print('\terror is\t\t\t',err)
		for i in range(len(self.we)-1,-1,-1):
			grad_w=self.lr*np.dot((err*self.dsigmoid(fin_list[i+1])),np.transpose(fin_list[i]))
			self.we[i]=self.we[i]+grad_w
			grad_b=self.lr*(err*self.dsigmoid(fin_list[i+1]))
			self.biases[i]=self.biases[i]+grad_b
			err=np.dot(np.transpose(self.we[i]),err)
		####training completes####
import random
def AND(val):
	a=[]
	b=[]
	if val==0:
		print('\t\t\tAND gate')
	for i in range(2):
		a.append(int(random.choice('10')))
	if a[0]==1 and a[1]==1:
		b.append(1)
	else:
		b.append(0)
	return a,b
	
def OR(val):
	a=[]
	b=[]
	if val==0:
		print('\t\t\tOR gate')
	for i in range(2):
		a.append(int(random.choice('10')))
	if a[0]==1 or a[1]==1:
		b.append(1)
	else:
		b.append(0)
	return a,b
	
def XOR(val):
	a=[]
	b=[]
	if val==0:
		print('\t\t\tXOR gate')
	for i in range(2):
		a.append(int(random.choice('10')))
	if a[0] != a[1]:
		b.append(1)
	else:
		b.append(0)
	return a,b
	
def XNOR(val):
	a=[]
	b=[]
	if val==0:
		print('\t\t\tXNOR gate')
	for i in range(2):
		a.append(int(random.choice('10')))
	if a[0]==a[1]:
		b.append(1)
	else:
		b.append(0)
	return a,b
	
hid=[]
inp=int(input('enter no of hidden layers=>'))
for i in range(inp):
	hid.append(int(input('how many nodes in h'+str(i+1)+'=>')))
l=[2]
l.extend(hid)
l.append(1)
epochs=10000
obj=network(l)
print('Enter 1 for AND\nEnter 2 for OR\nEnter 3 for XOR\nEnter 4 for XNOR')
bt=''
while bt!='1' or bt!='2' or bt!='3' or bt!='4':
	bt=input('enter your choice=>')
	if bt=='1' or bt=='2' or bt=='3' or bt=='4':
		break

bt=int(bt)
val=0
for i in range(epochs):
	it=i
	if bt==1:
		a,b=AND(val)
		obj.train(a,b,it)
		val=1
		
	elif bt==2:
		a,b=OR(val)
		obj.train(a,b,it)
		val=1
	
	elif bt==3:
		a,b=XOR(val)
		obj.train(a,b,it)
		val=1
		
	elif bt==4:
		a,b=XNOR(val)
		obj.train(a,b,it)
		val=1
	
ch=1
a=''
while ch==1:
	print('enter the value one by one')
	while a!='1' or a!='0':
		a=input('val1=>')
		if a=='1' or a=='0':
			break
	a=int(a)
	while b!='1' or b!='0':
		b=input('val2=>')
		if b=='1' or b=='0':
			break
	b=int(b)
	fin=obj.query([a,b])
	print([a,b],'=>',fin)
	print('Do you want to continue??\nEnter 1 to continue and 0 to exit')
	ch=int(input('enter please=>'))
