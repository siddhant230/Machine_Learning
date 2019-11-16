import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
from scipy.spatial import distance
import time
warnings.filterwarnings('ignore')

def ann():
    class network:
        def __init__(self,inp_list):
            self.we=[]
            self.err=[]
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
            self.lr=0.01

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

        def cost(self):
            #print('\terror is\t\t\t',self.err.shape)
            pass

        #######train########
        def train(self,input,target):
            fin_list=[]
            ip=np.array(input)
            target=np.array(target).reshape(4,1)
            val=np.reshape(ip,(self.inp[0],1))

            for i in range(len(self.inp)-1):
                fin_list.append(val)
                z=np.dot(self.we[i],val)+self.biases[i]
                a=self.sigmoid(z)
                val=a
            fin_list.append(val)
            out=fin_list[-1]
            self.err=target-out
            #print(out.shape,target.shape,self.err.shape)
            for i in range(len(self.we)-1,-1,-1):
                grad_w=self.lr*np.dot((self.err*self.dsigmoid(fin_list[i+1])),np.transpose(fin_list[i]))
                self.we[i]=self.we[i]+grad_w
                grad_b=self.lr*(self.err*self.dsigmoid(fin_list[i+1]))
                self.biases[i]=self.biases[i]+grad_b
                self.err=np.dot(np.transpose(self.we[i]),self.err)
            ####training completes####
    hid=[]
    inp=int(input('enter no of hidden layers=>'))
    for i in range(inp):
        hid.append(int(input('how many nodes in h'+str(i+1)+'=>')))
    l=[5]
    l.extend(hid)
    l.append(4)
    epochs=50
    obj=network(l)

    import pandas as pd
    from sklearn.utils import shuffle
    df=pd.read_csv('C:\\Users\\tusha\Desktop\\final.csv')
    df=shuffle(df)
    df=df.head(50000)
    target=df['target']
    target=pd.get_dummies(target)
    df=df.drop('target',axis=1)

    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    std=StandardScaler()
    df=std.fit_transform(df)

    from sklearn.model_selection import train_test_split
    xtr,xte,ytr,yte=train_test_split(df,target,test_size=0.2)

    it=0
    ytr=np.array(ytr)
    for _ in tqdm(range(epochs)):
        for x,y in zip(xtr,ytr):
            obj.train(x,y)

    pre=[]
    op=[1,2,3,4]
    for x in xte:
        val=obj.query(x)
        val=[i[0] for i in val]
        pre.append(op[val.index(max(val))])
    c=0
    print(pre)
    yte=yte.idxmax(axis=1)
    print("PREDICTION : {}".format(yte))
    from sklearn.metrics import accuracy_score
    print("ACCURACY SCORE {}".format(accuracy_score(pre,yte)))

def svm():
    print('LOADING DATA...')
    df=pd.read_csv('C:\\Users\\tusha\Desktop\\final.csv')
    df=shuffle(df)
    df=df.head(20000)
    target=df['target']

    df=df.drop('target',axis=1)
    print('TRANSFORMING...')
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    df=std.fit_transform(df)

    print('SPLITTING THE DATA...')
    from sklearn.model_selection import train_test_split
    xtr,xte,ytr,yte=train_test_split(df,target,test_size=0.2)
    ytr=np.array(ytr)

    print('TRAINING AND FITTING THE MODEL...')
    ##SUPPORT VECTOR MACHINE
    c_r=10
    from sklearn.svm import SVC
    for c in range(1,c_r+1):
        obj=SVC(C=c)
        obj.fit(xtr,ytr)
        pre=obj.predict(xte)
        print('PREDICTION : for c={}.0'.format(c))
        print("predicion is {}".format(pre))
        from sklearn.metrics import accuracy_score
        print("Accuracy Score : {}".format(accuracy_score(pre,yte)))

def som():
    print('LOADING DATA...')
    df=pd.read_csv('C:\\Users\\tusha\Desktop\\final.csv')
    df=shuffle(df)
    df=df.head(2500)
    target=df['target']

    df=df.drop('target',axis=1)
    print('TRANSFORMING...')
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    std=StandardScaler()
    df=std.fit_transform(df)

    print('SPLITTING THE DATA...')
    from sklearn.model_selection import train_test_split
    xtr,xte,ytr,yte=train_test_split(df,target,test_size=0.2)
    ytr=np.array(ytr)

    ###weight neuron construction
    print('TRAINING AND FITTING THE MODEL...')
    weight=np.random.uniform(low=0,high=1,size=(xtr[0].shape)+(4,)).reshape((4,5))
    lr=0.1
    epochs=20
    for e in tqdm(range(epochs)):
        for r in range(len(xtr)):
            deltas=[]

            ##finding deltas
            for w in weight:
                delta=distance.euclidean(xtr[r],w)
                deltas.append(delta)
            index_of_max_weighted_neuron=deltas.index(max(deltas))

            ##weight updation
            for ind in range(len(xtr[r])):
                weight[index_of_max_weighted_neuron][ind]=weight[index_of_max_weighted_neuron][ind] + lr*(xtr[r][ind]-weight[index_of_max_weighted_neuron][ind])
    print('UPDATED WEIGHT NEURON')
    print(weight)

def lvq():
    print('LOADING DATA...')
    df=pd.read_csv('C:\\Users\\tusha\Desktop\\final.csv')
    df=shuffle(df)
    df=df.head(10000)
    target=df['target']
    df=df.drop('target',axis=1)

    print('TRANSFORMING ...')
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    std=StandardScaler()
    df=std.fit_transform(df)

    print('SPLITTING DATA...')
    from sklearn.model_selection import train_test_split
    xtr,xte,ytr,yte=train_test_split(df,target,test_size=0.2)
    ytr=np.array(ytr)

    ###weight neuron construction
    print('TRAINING DATA...')
    weight=np.random.uniform(low=0,high=1,size=(xtr[0].shape)+(4,)).reshape((4,5))
    print('INITIAL WEIGHT...')
    print(weight)
    time.sleep(2)
    lr=1e-5
    epochs=5
    for e in tqdm(range(epochs)):
        for r in range(len(xtr)):
            deltas=[]

            ##finding deltas
            for w in weight:
                delta=distance.euclidean(xtr[r],w)
                deltas.append(delta)
            index_of_max_weighted_neuron=deltas.index(max(deltas))

            ##weight updation
            for w_i in range(len(weight)):
                for ind in range(len(xtr[r])):
                    if w_i==index_of_max_weighted_neuron:
                        weight[index_of_max_weighted_neuron][ind]=weight[index_of_max_weighted_neuron][ind] + lr*(xtr[r][ind]-weight[index_of_max_weighted_neuron][ind])
                    else:
                        weight[w_i][ind]=weight[w_i][ind] - lr*(xtr[r][ind]-weight[w_i][ind])
    time.sleep(1)
    print('WEIGHT AFTER UPDATION...')
    print(weight)

if __name__=="__main__":
    print('ENTER 1 FOR ANN\nENTER 2 FOR SVM\nENTER 3 FOR SOM\nENTER 4 FOR LVQ')
    choice=int(input('ENTER CHOICE : '))
    if choice==1:
        ann()
    if choice==2:
        svm()
    if choice==3:
        som()
    if choice==4:
        lvq()
