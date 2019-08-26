#왜안되냐
import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("Logit_layer"):
  W2 = tf.Variable(tf.random_uniform ([1], -100., 100.))
  hypothesis = W2 * X# + b
  
with tf.name_scope("GD_Trainer"):
  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  rate = tf.Variable(0.1)
  gradient = tf.reduce_mean((W2 * X -Y) * X)
  desent = W2 - rate * gradient
  update = tf.assign(W2,desent)
  #optimizer = tf.train.GradientDescentOptimizer(rate)
  #train = optimizer.minimize (cost )
  
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter('/content/log', graph = tf.get_default_graph())


#여기서부터 train
for step in range(2001):
  sess.run(train, feed_dict = {X:x_data, Y:y_data})
  if step% 20 == 0:
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
    

print(sess.run(hypothesis, feed_dict={X : 5}))
print(sess.run(hypothesis, feed_dict={X : 2.5}))
print(sess.run(hypothesis, feed_dict={X: [2.5, 5]}))