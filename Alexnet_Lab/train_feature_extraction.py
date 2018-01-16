import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle



nb_classes = 43
epoch = 10
batch_size = 128

# TODO: Load traffic signs data.
with open('./train.p','rb') as f:
	data = pickle.load(f)

# TODO: Split data into training and validation sets.
# Pay attention to the order X_train,X_val,y_train,y_val
X_train, X_val, y_train, y_val = train_test_split(data['features'][:1000],data['labels'][:1000],test_size = 0.33, random_state = 0)

# TODO: Define placeholders and resize operation.
feature = tf.placeholder(tf.float32,(None,32,32,3))
label = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(feature,(227,227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1],nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape,stddev = 1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7,fc8W,fc8b)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = label)
# spare_softmax_cross_entropy_with_logits applies when If you have single-class labels, where an object can only belong to one class
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list = [fc8W,fc8b])
init_op = tf.global_variables_initializer()

preds = tf.argmax(logits,1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds,label),tf.float32))

# TODO: Train and evaluate the feature extraction model.
# Define model evaluation
def evaluate(X_data,y_data,sess):
	num_examples = len(X_data)
	#sess = tf.get_default_session()
	total_loss, total_acc = 0,0
	for offset in range(0,num_examples,batch_size):
		end = offset+batch_size
		batch_x, batch_y = X_data[offset:end], y_data[offset:end]
		loss, acc = sess.run([loss_op, accuracy_op],feed_dict={feature:batch_x,label:batch_y})
		total_loss += loss*batch_size
		total_loss += acc*batch_size
	total_loss /= num_examples
	total_acc /= num_examples
	return total_loss,total_acc

# Train model here
valid_eval = []

with tf.Session() as sess:
	sess.run(init_op)
	num_examples = len(X_train)
	for i in range(epoch):
		print('start training for EPOCH-{}'.format(i))
		X_train,y_train = shuffle(X_train,y_train)

		for offset in range(0,num_examples,batch_size):
			end = offset+batch_size
			batch_x, batch_y = X_train[offset:end], y_train[offset:end]
			sess.run(train_op,feed_dict={feature:batch_x,label:batch_y})

		#train_loss,train_acc = evaluate(X_train,y_train)
		valid_loss,valid_acc = evaluate(X_val,y_val,sess)
		valid_eval.append([valid_loss,valid_acc])
		print('for EPOCH-{}'.format(i))
		print('Validation loss = ',valid_loss)
		print('Validation accuracy = ',valid_acc)


