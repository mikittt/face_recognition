import cPickle
import numpy as np
import os
from PIL import Image
import argparse
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers, serializers ### Add 'serializers'
import chainer.functions  as F
import chainer.links as L
import pdb
import argparse
from VGGNet import VGGNet
from Original_VGGNet import Original_VGGNet
import glob
from sklearn.cross_validation import train_test_split
import cPickle as pickle
import cv2
#parser = argparse.ArgumentParser()
#parser.add_argument('--gpu',default=0,help='GPU ID (default 0)')
#parser.add_argument('--model',default=0,help='use pre-trained model or not')
#parser.add_argument('--load',default=1,help='LOAD or not (default 1)')
#args = parser.parse_args()


classes =  ["aragaki_face","hoshino_face","narita_face","other_face","fujii_face","mano_face","ohtani_face","yamaga_face"]
#load=int(args.load)


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def conv_setup(ORIGINAL_VGG,VGG):
    VGG.conv1_1 = ORIGINAL_VGG.conv1_1
    VGG.conv1_2 = ORIGINAL_VGG.conv1_2
    VGG.conv2_1 = ORIGINAL_VGG.conv2_1
    VGG.conv2_2 = ORIGINAL_VGG.conv2_2
    VGG.conv3_1 = ORIGINAL_VGG.conv3_1
    VGG.conv3_2 = ORIGINAL_VGG.conv3_2
    VGG.conv3_3 = ORIGINAL_VGG.conv3_3
    VGG.conv4_1 = ORIGINAL_VGG.conv4_1
    VGG.conv4_2 = ORIGINAL_VGG.conv4_2
    VGG.conv4_3 = ORIGINAL_VGG.conv4_3
    VGG.conv5_1 = ORIGINAL_VGG.conv5_1
    VGG.conv5_2 = ORIGINAL_VGG.conv5_2
    VGG.conv5_3 = ORIGINAL_VGG.conv5_3
    return VGG


def preprocess(x_train, x_test):
    print(x_train.shape)
    mean = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
    return x_train-mean, x_test-mean

def augment(x,size):#4D tensor
    #flip = np.random.randint(2,size = len(x))*2-1
    theta = np.random.uniform(0,2*np.pi,len(x))
    scale = np.random.uniform(0.9,1.2,len(x)).reshape(-1,1,1)
    shift = np.random.uniform(-5,5,len(x)*2).reshape(-1,2)
    xs = np.arange(size**2)%size
    ys = np.arange(size**2)/size
    coords = np.c_[xs,ys].transpose()-size/2.#変換前の座標
    R = scale*(np.c_[np.cos(theta),-np.sin(theta),np.sin(theta),np.cos(theta)].reshape(len(x),2,2))
    img = np.array([X[:,np.clip(np.dot(r,coords)[1]+size/2.+s[1],0,size-1).astype('int32'),np.clip(np.dot(r,coords)[0]+size/2.+s[0],0,size-1).astype('int32')].reshape(3,size,size) for s,r,X in zip(shift,R,x)])
    return img

def load_faces():
    data_dir = "/home/mil/noguchi/seminar/face/faces_nparray/"
    classes =  ["aragaki_face","hoshino_face","narita_face","other_face","fujii_face","mano_face","ohtani_face","yamaga_face"]
    X=[]
    y=[]
    
    for i, cls in enumerate(classes):
        X_=np.load(data_dir+cls+".npy").astype(np.uint8)
        kkk=np.array([cv2.resize(ll,((96,96))).transpose(2,0,1)/255. for ll in X_])
        X.extend(kkk)
        y.extend(np.ones(len(kkk),dtype=np.int32)*i)
    X = np.array(X,dtype=np.float32)
    y = np.array(y,dtype=np.int32)

    return X,y

def valid_train():
    with open('vgg.pkl', 'rb') as i:
        orig_vgg = pickle.load(i)
    vgg=VGGNet()
    #if int(args.model) == 1:
    #orig_vgg = Original_VGGNet()
    #serializers.load_hdf5('VGG.model', orig_vgg)
    vgg = conv_setup(orig_vgg,vgg)
    print 'loading data now.'
    
    #if not load :
    X,y = load_faces()
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print 'loading data done'
    x_train,x_test = preprocess(x_train,x_test)
    batchsize = 30
    N = len(x_train)
    N_test = len(x_test)
    n_epoch = 50
    #gpu = int(args.gpu)
    optimizer = optimizers.MomentumSGD(lr=0.002,momentum=0.9)
    
    optimizer.setup(vgg)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
    vgg.to_gpu()
    
    for epoch in xrange(1, n_epoch+1):
        
        if epoch in [35,45]:
            optimizer.lr*=0.20
        print 'epoch', epoch
        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in xrange(0, N, batchsize):
            x_batch = cuda.to_gpu(augment(x_train[perm[i:i+batchsize]],96))
            y_batch = cuda.to_gpu(y_train[perm[i:i+batchsize]])
            optimizer.zero_grads()
            loss,_,_ = vgg(x_batch, y_batch, train=True)
            loss.backward()
            optimizer.update()
            sum_loss     += float(cuda.to_cpu(loss.data)) * len(x_batch)

        print 'train mean loss={}'.format(sum_loss / N)
        
        # evaluation
        sum_accuracy = 0
        
        pred_y =[]
        for i in xrange(0, N_test, batchsize):
            x_batch = cuda.to_gpu(x_test[i:i+batchsize])
            y_batch = cuda.to_gpu(y_test[i:i+batchsize])

            _,acc,pred = vgg(x_batch,y_batch,train=False)
            pred_y.extend(np.argmax(cuda.to_cpu(pred.data),axis=1))
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(x_batch)

        print 'test mean accuracy={}'.format(sum_accuracy / N_test)
        for i in range(len(classes)):
            accuracy = np.sum((np.array(pred_y)==y_test)*(y_test==i))*1./np.sum(y_test==i)
            print '    {} accuracy={}'.format(classes[i],accuracy)

def train():
    with open('vgg.pkl', 'rb') as i:
        orig_vgg = pickle.load(i)
    vgg=VGGNet()
    #if int(args.model) == 1:
    #orig_vgg = Original_VGGNet()
    #serializers.load_hdf5('VGG.model', orig_vgg)
    vgg = conv_setup(orig_vgg,vgg)
    print 'loading data now.'
    
    #if not load :
    X,y_train = load_faces()
    
    print 'loading data done'
    x_train,_= preprocess(X,[])
    batchsize = 30
    N = len(x_train)
    n_epoch = 40
    #gpu = int(args.gpu)
    optimizer = optimizers.MomentumSGD(lr=0.002,momentum=0.9)
    
    optimizer.setup(vgg)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
    vgg.to_gpu()
    
    for epoch in xrange(1, n_epoch+1):
        
        if epoch in [30,35]:
            optimizer.lr*=0.1
        print 'epoch', epoch
        # training
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in xrange(0, N, batchsize):
            x_batch = cuda.to_gpu(augment(x_train[perm[i:i+batchsize]],96))
            y_batch = cuda.to_gpu(y_train[perm[i:i+batchsize]])
            optimizer.zero_grads()
            loss,_,_ = vgg(x_batch, y_batch, train=True)
            loss.backward()
            optimizer.update()
            sum_loss     += float(cuda.to_cpu(loss.data)) * len(x_batch)

        print 'train mean loss={}'.format(sum_loss / N)
        
    serializers.save_hdf5('VGG11_{}.model'.format(str(sum_loss/N).replace('.','')), vgg)


if __name__ =='__main__':
    valid_train()
