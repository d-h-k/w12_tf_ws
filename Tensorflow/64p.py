import tensorflow as tf
import numpy as np

xy = np.loadtxt('./softmax.txt', unpack=True, dtype = 'float32')
x_data = np
y_data = 

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
h = tf.matmul(W,X)

hypothesis = tf.div(1.0 , 1. + tf.exp(-h))
cost = -tf.reduce_mean(-tf.reduce@@@
                       
#rate = tf.Variable(0.1)
learning_rate = .01                       
    
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
#cf..

with tf.Session() as sess:                                              
  sess.run(init)
  for step in range(2001):
        sess.run(train, feed_dict = {X:x_data, Y:y_data})
        if step %200 == 0:
            print('{:4}{:8.6}'.format(step,
            sess.run(cost, {x:x_data, Y:y_data}), sess.run(W))
        
  print('test','-'*50)

  a = sess.run(hypothesis, feed_dict = {X:[[1,11,7]]})
  print('[1,11,7]:',a, sess.run(tf.argmax(a,1)))
  
  b = sess.run(hypothesis, feed_dict = {X: [[1,3,4]]})  
  print('[1,3,4]:',b, sess.run(tf.argmax(b,1)))


  c = sess.run(hypothesis, feed_dict = {X: [[1,1,0]]})  
  print('[1,1,0]:',c, sess.run(tf.argmax(c,1)))

  d = sess.run(hypothesis, feed_dict = {X: [[1,3,4]]})  
  print(sess.run(tf.argmax(d,1)))
  #print('[1,11,7], [1,3,4], [1,5,5], grade?',d, sess.run(tf.argmax(b,1)))
  
    
  print('[1,5,5]:',sess.run(hypothesis, feed_dict = {X:[[1],[5],[5]]}) )

print(sess.run(hypothesis, feed_dict = {X:[[1,1],[2,5],[2,5]]})>0.5)
sess.close()



