class KNeighborsClassifiers():
    
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y
        
    def predict(self,x_test):
        pre=[]
        for i in x_test:
            m=self.closest(i)
            pre.append(m)
        return pre
    
    def closest(self,row):
        index=0
        dist=0
        best=self.euclid(row,self.x_train[0])
        for i in range(len(self.x_train)):
            dist=self.euclid(row,self.x_train[i])
            if(dist<best):
                best=dist
                index=i
        return self.y_train[index]
    
    def euclid(self,a,b):
        add=0
        for i in range(4):
            diff=a[i]-b[i]
            sq=diff**2
            add=add+sq
        sq_root=add**0.5
        return sq_root

from sklearn.datasets import load_iris
import pandas as pd
iris=load_iris()

x=iris.data
y=iris.target
a,b,c,d=[],[],[],[]
for i in range(len(x)):
	a.append(x[i][0])
	b.append(x[i][1])
	c.append(x[i][2])
	d.append(x[i][3])

df=pd.DataFrame(x,columns=['a','b','c','d'])
print(df.isnull().sum())


from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
n_size=0.3
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=n_size)
print(len(x_train),len(x_test))
sequence=[x for x in range(1,len(x_test))]
def distributer(sequence,x_train,x_test,y_train,y_test):
    index=0
    best=9999
    for i in sequence:
        clf=KNeighborsClassifiers()
        clf.fit(x_train,y_train)
        pre=clf.predict(x_test)
        mae= mean_absolute_error(y_test,pre)
        if(best>mae):
            best=mae
            index=i
    return (index/100)
        
best_distribution=distributer(sequence,x_train,x_test,y_train,y_test)
print(best_distribution)
best_distribution=0.3
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=best_distribution)
clf=KNeighborsClassifiers()
mod=KNeighborsClassifier(n_neighbors=10)
mod.fit(x_train,y_train)
pr=mod.predict(x_test)

clf.fit(x_train,y_train)
pre=clf.predict(x_test)
for i in pre:
    print("prediction is",iris.target_names[i])

for j in y_test:
    print("expected is",iris.target_names[j])
    
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pre)*100)
print(accuracy_score(y_test,pr)*100)