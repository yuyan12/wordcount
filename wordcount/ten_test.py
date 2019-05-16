# from django.http import HttpResponse

import tensorflow as tf
import sys
sys.path.append('./model/')
import single_judge


'''
en1_1 = 'A'
en2_1 = 'B'
text = 'AandB'
ans = 'null'
ans = single_judge.single_judge(en1_1, en2_1, text)
#sess = tf.Session()

'''
def main1(en1_1,en2_1,text):
    ans = 'null'
    ans = single_judge.single_judge(en1_1, en2_1, text)

main1('A','B','AdfjskldfjklB')

'''


if __name__ == "__main__":
    tf.app.run()
'''