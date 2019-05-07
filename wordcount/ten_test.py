# from django.http import HttpResponse

import tensorflow as tf
import sys
sys.path.append('./model/')
import single_judge



def main(_):
    en1_1 = 'A'
    en2_1 = 'B'
    text = 'AandB'
    ans = 'null'
    ans = single_judge.single_judge(en1_1, en2_1, text)
    print(ans)


if __name__ == "__main__":
    tf.app.run()