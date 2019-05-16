# from django.http import HttpResponse
from django.shortcuts import render  # 传网页给用户

import tensorflow as tf
import numpy as np
#import network

# FLAGS = tf.app.flags.FLAGS

def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


class Settings(object):
    def __init__(self):
        self.vocab_size = 16691
        self.num_steps = 70
        self.num_epochs = 10
        self.num_classes = 12
        self.gru_size = 230
        self.keep_prob = 0.5
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50


class GRU:
    def __init__(self, is_training, word_embeddings, settings):

        self.num_steps = num_steps = settings.num_steps
        self.vocab_size = vocab_size = settings.vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.gru_size = gru_size = settings.gru_size
        self.big_num = big_num = settings.big_num

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[big_num + 1], name='total_shape')
        total_num = self.total_shape[-1]

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [settings.pos_num, settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [settings.pos_num, settings.pos_size])

        attention_w = tf.get_variable('attention_omega', [gru_size, 1])
        sen_a = tf.get_variable('attention_A', [gru_size])
        sen_r = tf.get_variable('query_r', [gru_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, gru_size])
        sen_d = tf.get_variable('bias_d', [self.num_classes])

        gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

        if is_training and settings.keep_prob < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * settings.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * settings.num_layers)

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

        self._initial_state_forward = cell_forward.zero_state(total_num, tf.float32)
        self._initial_state_backward = cell_backward.zero_state(total_num, tf.float32)

        # embedding layer
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)])
        inputs_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos2_embedding,
                                                                   tf.reverse(self.input_pos2, [1]))])

        outputs_forward = []

        state_forward = self._initial_state_forward

        # Bi-GRU layer
        with tf.variable_scope('GRU_FORWARD') as scope:
            for step in range(num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
                outputs_forward.append(cell_output_forward)

        outputs_backward = []

        state_backward = self._initial_state_backward
        with tf.variable_scope('GRU_BACKWARD') as scope:
            for step in range(num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
                outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [total_num, num_steps, gru_size])
        output_backward = tf.reverse(
            tf.reshape(tf.concat(axis=1, values=outputs_backward), [total_num, num_steps, gru_size]),
            [1])

        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
                       [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])

        # sentence-level attention layer
        for i in range(big_num):

            sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [gru_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.num_classes]), sen_d))

            self.prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        # tf.summary.scalar('loss',self.total_loss)
        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)

#=========================single_judge============================
def single_judge(en1, en2, sentence):
    # If you retrain the model,
    # please remember to change the path to your own model below:
    pathname = "./modelA/ATT_GRU_model-3500"

    wordembedding = np.load('C:/Users/C/Desktop/Git_re/wordcount/wordcount/vec.npy')
    test_settings = Settings()
    test_settings.vocab_size = 16693
    test_settings.num_classes = 11
    test_settings.big_num = 1

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            with tf.variable_scope("modelA"):
                mtest = GRU(is_training=False,
                    word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            print('reading word embedding data...')
            vec = []
            word2id = {}
            f = open('./origin_dataA/vec.txt', encoding='utf-8')
            content = f.readline()
            content = content.strip().split()
            dim = int(content[1])
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)
            f.close()
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            print('reading relation to id')
            relation2id = {}
            id2relation = {}
            f = open('./origin_dataA/relation2id.txt', 'r', encoding='utf-8')
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])
                id2relation[int(content[1])] = content[0]
            f.close()

            try:
                # BUG: Encoding error if user input directly from command line.
                # line = input('请输入中文句子 "name1 name2 sentence":')
                # Read file from test file

                # en1, en2, sentence = line.strip().split()
                print("实体1: " + en1)
                print("实体2: " + en2)
                print(sentence)
                relation = 0
                en1pos = sentence.find(en1)
                if en1pos == -1:
                    en1pos = 0
                en2pos = sentence.find(en2)
                if en2pos == -1:
                    en2post = 0
                output = []
                # length of sentence is 70
                fixlen = 70
                # max length of position embedding is 60 (-60~+60)
                maxlen = 60

                # Encoding test x
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = pos_embed(i - en1pos)
                    rel_e2 = pos_embed(i - en2pos)
                    output.append([word, rel_e1, rel_e2])

                for i in range(min(fixlen, len(sentence))):

                    word = 0
                    if sentence[i] not in word2id:
                        # print(sentence[i])
                        # print('==')
                        word = word2id['UNK']
                        # print(word)
                    else:
                        # print(sentence[i])
                        # print('||')
                        word = word2id[sentence[i]]
                        # print(word)

                    output[i][0] = word
                test_x = []
                test_x.append([output])

                # Encoding test y
                label = [0 for i in range(len(relation2id))]
                label[0] = 1
                test_y = []
                test_y.append(label)

                test_x = np.array(test_x)
                test_y = np.array(test_y)

                test_word = []
                test_pos1 = []
                test_pos2 = []

                for i in range(len(test_x)):
                    word = []
                    pos1 = []
                    pos2 = []
                    for j in test_x[i]:
                        temp_word = []
                        temp_pos1 = []
                        temp_pos2 = []
                        for k in j:
                            temp_word.append(k[0])
                            temp_pos1.append(k[1])
                            temp_pos2.append(k[2])
                        word.append(temp_word)
                        pos1.append(temp_pos1)
                        pos2.append(temp_pos2)
                    test_word.append(word)
                    test_pos1.append(pos1)
                    test_pos2.append(pos2)

                test_word = np.array(test_word)
                test_pos1 = np.array(test_pos1)
                test_pos2 = np.array(test_pos2)

                prob, accuracy = test_step(test_word, test_pos1, test_pos2, test_y)
                prob = np.reshape(np.array(prob), (1, test_settings.num_classes))[0]
                print("关系是:")
                # print(prob)
                top3_id = prob.argsort()[-3:][::-1]
                for n, rel_id in enumerate(top3_id):
                    ans = "No." + str(n+1) + ": " + id2relation[rel_id] + ", Probability is " + str(prob[rel_id])
                    print("No." + str(n+1) + ": " + id2relation[rel_id] + ", Probability is " + str(prob[rel_id]))
                return ans
            except Exception as e:
                print(e)
#=================================================================

def home(request):
    return render(request, 'home.html')


def count(request):
    total_count = len(request.GET['text'])  # text是字典形式的
    user_text = request.GET['text']
    en1_1 = request.GET['en1']
    en2_1 = request.GET['en2']
    ans = 'null'
    print('---------------------count---------------------')
    #ans = single_judge(en1_1, en2_1, user_text) //这里改一下试试，变量要改的
    #ans = single_judge('A', 'B', 'AandB')
    print('---------------------count1---------------------')
    return render(request, 'count.html', {'count': total_count,
    	'text': user_text, 'en1': en1_1, 'en2': en2_1, 'relation': ans})
    # render可以向html中传递信息，使用字典的方式

def about(request):
    return render(request, 'about.html')