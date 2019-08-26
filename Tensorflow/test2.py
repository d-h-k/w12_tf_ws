# 왜 되냐??
import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("Logit_Layer"):
  W2 = tf.Variable(tf.random_uniform ([1], -100., 100.))
  hypothesis = W2 * X # + b
  
with tf.name_scope("GD_Trainer"):
  cost = tf.reduce_mean(tf.square(hypothesis - Y))  #cost
  rate = tf.Variable(0.1)
  gradient = tf.reduce_mean((W2 * X - Y) * X)    #gradient 실습하기 위해 추가
  descent = W2 - rate * gradient   #alpha 는 Learning rate~  # descent 
  update = tf.assign(W2, descent)
#   optimizer = tf.train.GradientDescentOptimizer(rate)
#   train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
#cf.)tf.initialize_all_variables() using in the old version

sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter('/content/log',graph=tf.get_default_graph())

#여기서부터 train
for step in range(200):
#   sess.run(train, feed_dict={X: x_data, Y: y_data})
  if step% 5 == 0:
    print('step:', step, sess.run(update, feed_dict={X: x_data, Y: y_data}),'W2:',sess.run(W2))

#여기서부터 test
#관측되지 않은 데이터 넣어보기 5, 2.5
print(sess.run(hypothesis, feed_dict={X : 5}))    
print(sess.run(hypothesis, feed_dict={X : 2.5}))
print(sess.run(hypothesis, feed_dict={X: [2.5, 5]}))