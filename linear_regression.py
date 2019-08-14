import numpy as np
class regression:	
	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.lr=0.0001
		self.m=np.zeros((1,1),dtype='float')
		self.b=np.zeros((1,1),dtype='float')
		self.iters=1000
	
	def calc_cost(self):
		total=0.0
		for i in range(len(self.x)):
			y=self.y[i]
			x=self.x[i]
			total+=(y-(np.dot(self.m,x)+self.b))**2
		print(total/float(len(self.x)),self.m,self.b)
			
	def step_grad(self):
		grad_b=0
		grad_m=0
		n=float(len(self.x))
		for i in range(len(self.x)):
			y=self.y[i]
			x=self.x[i]
			grad_b+=-(2/n)*(y-(np.dot(self.m,x)+self.b))
			grad_m+=-(2/n)*x*(y-(np.dot(self.m,x)+self.b))
			#grad_b+=-(y-(np.dot(self.m,x)+self.b))
			#grad_m+=-x*(y-(np.dot(self.m,x)+self.b))
		self.b=self.b-(self.lr*grad_b)
		self.m=self.m-(self.lr*grad_m)
		return self.b,self.m
			
	def main_grad_runner(self):
		for i in range(self.iters):
			self.b,self.m=self.step_grad()
			if i%10==0:
				self.calc_cost()

if __name__ == '__main__':
	import  pandas as pd
	import matplotlib.pyplot as plt
	obj=pd.read_csv('data.csv')
	train_x,train_y=obj['a'],obj['b']
	train_x=np.array([1,2,3,4,5])
	train_y=np.array([2,4,6,8,10])
	ob=regression(train_x,train_y)
	ob.main_grad_runner()
	error=ob.calc_cost()
	print('#########')
	print('m is ',ob.m[0][0],'\nb is ',ob.b[0][0])
	m=ob.m[0][0]
	b=ob.b[0][0]
	x_ax=[i for i in range(100)]
	y_ax=[]
	for i in x_ax:
		y=m*i+b
		y_ax.append(y)
	plt.scatter(train_x,train_y,c='r')
	plt.plot(x_ax,y_ax)
	b=b+10
	
	pari_x=[1,2,3,4,5]
	pari_y=[m*x+b for x in pari_x]
	plt.scatter(pari_x,pari_y)
	plt.savefig('linear')
