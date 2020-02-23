from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

iteration = 250
mnist= input_data.read_data_sets('MNIST_data/',one_hot=True)

train=mnist.train.images[:20000]
train_labels=mnist.train.labels[:20000]

test=mnist.test.images
test_labels=mnist.test.labels

t_arr=np.arange(51)/float(50)
def learner(x,t):
    u_arr=dict()
    u=x>=t
    upos=2*u-1
    uneg=upos*-1
    u_arr['pos']=upos
    u_arr['neg']=uneg
    return u_arr
def alpha_test(j,polarity,test,limit):
    u=test[:,j]>=limit
    u=2*u-1
    if polarity==-1:
       u=u*-1
    u=np.reshape(u,(1,test.shape[0]))
    return u
def cal_error(label,g):
    d=g>0
    d=2*d-1

    error=1-np.sum(d==label)/float(label.shape[1])
    return error
            
    
decision_stump=dict()
for i,t in enumerate(t_arr):
    decision_stump[i]=learner(train,t)
    
iter_track=[5,10,50,100,250]
def classifier(number):
    g=np.zeros((1,train.shape[0]))
    g_test=np.zeros((1,test.shape[0]))
    weights=np.zeros((1,train.shape[0]))
    weights=np.zeros((1,train.shape[0]))
    ytrain=train_labels[:,number]
    ytrain=2*ytrain-1
    ytest=test_labels[:,number]
    ytest=2*ytest-1
    ytrain=np.reshape(ytrain,(1,train.shape[0]))
    ytest=np.reshape(ytest,(1,test.shape[0]))
    #Initializations
    train_error=[]
    test_error=[]
    largest_weight=[]
    
    margin=[]
    array=128*np.ones((1,28*28))
    
    for i in range(iteration):
        weights=np.exp(-ytrain*g)
        grad_max=0
        for t in range(len(t_arr)):
            u=decision_stump[t]
            u_pos=u['pos']
            u_neg=u['neg']
            pos_grad=np.sum(np.multiply(np.multiply(ytrain,weights),u_pos.T),axis=1)
            pos_grad=np.reshape(pos_grad,(1,train.shape[1]))
            neg_grad=np.sum(np.multiply(np.multiply(ytrain,weights),u_neg.T),axis=1)
            neg_grad=np.reshape(neg_grad,(1,train.shape[1]))
           
            if grad_max>max(np.amax(pos_grad),np.amax(neg_grad)):
                continue
            grad_max=max(np.amax(pos_grad),np.amax(neg_grad))
            if np.amax(pos_grad)>=np.amax(neg_grad):
               j=np.argmax(pos_grad)
               alpha=u_pos[:,j]
               polarity=1
               limit=t_arr[t]
            else:
               j=np.argmax(neg_grad)
               alpha=u_neg[:,j]
               polarity=-1
               limit=t_arr[t]
        alpha=np.reshape(alpha,(1,train.shape[0]))  
        los=ytrain*alpha
        los=los<0
        loss=np.sum(np.multiply(los,weights),axis=1)/float(np.sum(weights,axis=1))
        step=0.5*np.log((1-loss[0])/float(loss[0]))
        g+=step*alpha
        g_test+=step*alpha_test(j,polarity,test,limit)
       
        train_error.append(cal_error(ytrain,g))
        test_error.append(cal_error(ytest,g_test))
        largest_weight.append(np.argmax(weights,axis=1))
        if i+1 in iter_track:
           margin.append(np.multiply(ytrain,g))
        #Qs 5d
        if polarity==1:
            array[0,j]=255
        else:
            array[0,j]=0
        
        features=dict()
        features['train_error']=train_error
        features['test_error']=test_error
        features['margin']=margin
        features['largest_weight']=largest_weight
        features['array']=array
    plt.figure(1)
    plt.plot(np.arange(1,iteration+1),train_error,label='train_dataset')
    
    plt.plot(np.arange(1,iteration+1),test_error,label='test_dataset')
    plt.xlabel('iteration')
    plt.ylabel('probability of error')
    plt.legend(loc='best')
    plt.show()
    n,h=np.histogram(margin[0],bins=10,normed=True)
   
    
    d=h[1]-h[0]
    cm=np.cumsum(n)*d
    plt.figure(2)
    plt.plot(h[1:],cm,label='iteration=5')
    n,h=np.histogram(margin[1],bins=10,normed=True)
    d=h[1]-h[0]
    cm=np.cumsum(n)*d
    plt.plot(h[1:],cm,label='iteration=10')
    n,h=np.histogram(margin[2],bins=10,normed=True)
    d=h[1]-h[0]
    cm=np.cumsum(n)*d
    plt.plot(h[1:],cm,label='iteration=50')
    n,h=np.histogram(margin[3],bins=10,normed=True)
    d=h[1]-h[0]
    cm=np.cumsum(n)*d
    plt.plot(h[1:],cm,label='iteration=100')
    n,h=np.histogram(margin[4],bins=10,normed=True)
    d=h[1]-h[0]
    cm=np.cumsum(n)*d
    plt.plot(h[1:],cm,label='iteration=250')
    plt.xlabel('iteration')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.show()
    plt.figure(3)
    plt.plot(np.arange(iteration),largest_weight)
    plt.xlabel('iteration')
    plt.ylabel('largest weight index')
    plt.show()
        
        
    (v,c)=np.unique(largest_weight,return_counts=True)
    ii=np.argmax(c)
    plt.figure(4)
    plt.imshow(np.reshape(train[v[ii]],(28,28)))
    c[ii]=0
    ii=np.argmax(c)
    plt.figure(5)
    plt.imshow(np.reshape(train[v[ii]],(28,28)))
    c[ii]=0
    ii=np.argmax(c)
    plt.figure(6)
    plt.imshow(np.reshape(train[v[ii]],(28,28)))
    plt.show()
    plt.figure(7)
    plt.imshow(np.reshape(array,(28,28)),cmap='gray')
    plt.show()
        
        
feature=dict()        
for digit in range(0,1):
    feature[digit]=classifier(digit)
    
        
    