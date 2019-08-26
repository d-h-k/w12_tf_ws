import tensorflow as tf
# Import MINST data

# Parameters.
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
                                  
# Set model weights
W1 = tf.Variable(tf.random_uniform([784, 256]) )
W2 = tf.Variable(tf.random_uniform([256, 256]))
W3 = tf.Variable(tf.random_uniform([256, 10]))

B1 = tf.Variable(tf.random_uniform([256]))
B2 = tf.Variable(tf.random_uniform([256]))
B3 = tf.Variable(tf.random_uniform([10]))

# Construct model
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2)) # Hidden layer with ReLU activation
hypothesis = tf.add (tf.matmul(L2, W3), B3)   # No need to use softmax here

# Minimize error using cross entropy
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels = Y))
# cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_#v2(hypothesis, Y))

# Gradient Descent
optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
                                    
# Initializing the variables
init = tf.global_variables_initializer()
#cf.)tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      # Run optimization op (backprop) and cost op (to get loss value)
      _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
      avg_cost += c / total_batch

      
      
      # Display logs per epoch step. display_step
    if (epoch+1) % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f} ".format(avg_cost))
  
  print("Optimization Finished!")

  # Test model
  correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
  
  # Calculate accuracy
  accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

  
  answers = sess.run(hypothesis, feed_dict={
      X: mnist.test.images, Y:mnist.test.labels
      })
  import numpy as np
  import matplotlib.pyplot as plt
  fig = plt.figure()
  for i in range(10):
    subplot = fig.add_subplot(2,5,i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(answers[i]))
    subplot.imshow(
    mnist.test.images[i].reshape([28,28]),
    cmap = plt.cm.gray_r)
plt.show()
    