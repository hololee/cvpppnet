import placeholders as holder
import tensorflow as tf

sum = holder.holder1 + holder.holder2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sum_result = sess.run(sum, feed_dict={holder.holder1: 1, holder.holder2: 3})

    print(sum_result)
