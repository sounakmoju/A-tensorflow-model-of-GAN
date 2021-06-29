import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.core import Dense,Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyRelu
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.python.keras.losses import binary_crossentropy

import numpy as np

def load_mnist():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train=x_train.Reshape(60000,784)
    return (x_train,y_train,x_test,y_test)
     #batch_count=x_train.shape[0]/batch_size
optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
random_dim=100

def build_generator(optimizer):
    generator=Sequential()
    generator.add(Dense(256,input_dim=random_dim,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyRelu(0.2))

    generator.add(Dense(512))
    generator.add(LeakyRelu(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyRelu(0.2))

    generator.add((Dense(784,activation='tanh')))
    generator.compile((loss='binary_crossentropy',optimizer=optimizer))
    return generator
def build_discriminator(optimizer):
    discriminator=Sequential()
    discriminator.add(Dense(1024,input_dim=784,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyRelu(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyRelu(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyRelu(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile((loss='binary_crossentropy',optimizer=optimizer))
    return discriminator


    generator.add((Dense(784,activation='tanh')))
    generator.compile((loss='binary_crossentropy',optimizer=optimizer))
def build_gan(dicriminator,random_dim,generator,optimizer):
    dicriminator.trainable=False
    gan_input=Input(shape=(random_dim,))
    x=generator(gan_input)
    gan_output=discriminator(x)
    gan=Model(inputs=gan_input,outputs=gan_output)
    gan.compile(loss='binary_crossentropy',optimizer)
    return gan
def train(epochs,batch_size):
    x_train,y_train,x_test,y_test=mnist.load_data
    iteration=x_train.shape[0]//batch_size
    generator=build_generator(optimizer)
    discriminator=build_discriminator(optimizer)
    gan=build_gan(dicriminator,random_dim,generator,optimizer)
    for i in range(1,epochs):
        for _ in tqdm(range(itaration)):
            noise=np.random.normal(0,1,size=[batch_size,random_dim])
            image_class=x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            image_gen=generator.predict(noise)
            X=np.concatenate([image_clas,image_gen])

            y_label=np.zeros(2*batch_size)
            y_label[:batch_size]=0.9
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_label)

            
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

            
    
    
                  
                      
    

