import numpy as np
import math
class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    def load_data(self, path):
        f_data = open(path , 'r')
        user_to_q_sequence = {}
        user_to_qa_sequence = {}
        user_id = 0
        q_data = []
        qa_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip( )
            if lineID % 3 == 0:
                user_id = line
            elif lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endIndex  = len(A)
                    else:
                        endIndex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endIndex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)

                q_dataArray = np.zeros((len(q_data), self.seqlen))
                for j in range(len(q_data)):
                    dat = q_data[j]
                    q_dataArray[j, :len(dat)] = dat
                user_to_q_sequence[user_id] = q_dataArray
                qa_dataArray = np.zeros((len(qa_data), self.seqlen))
                for j in range(len(qa_data)):
                    dat = qa_data[j]
                    qa_dataArray[j, :len(dat)] = dat
                user_to_qa_sequence[user_id] = qa_dataArray
                q_data.clear()
                qa_data.clear()
        f_data.close()
        return user_to_q_sequence, user_to_qa_sequence

    def load_test_data(self, path, test_set_rate):
        f_data = open(path , 'r')
        u2q_seq = {} # user to question sequence
        u2qa_seq = {}
        u2tf_seq = {} # tf: test flag
        Q = []
        A = []
        q_data = []
        qa_data = []
        tf_data = []
        user_id = 0
        train_seq_len = 0
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 3 == 0:
                user_id = line
            elif lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                train_seq_len = round(len(Q) * (1 - test_set_rate))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]

                n_split = 1
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    q_seq = []
                    a_seq = []
                    f_seq = []
                    if k == n_split - 1:
                        endIndex  = len(A)
                    else:
                        endIndex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endIndex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            q_seq.append(int(Q[i]))
                            a_seq.append(Xindex)
                            if i >= train_seq_len:
                                f_seq.append(Xindex)
                            else:
                                f_seq.append(0)
                        else:
                            print(Q[i])
                    q_data.append(q_seq)
                    qa_data.append(a_seq)
                    tf_data.append(f_seq)

                q_dataArray = np.zeros((len(q_data), self.seqlen))
                for j in range(len(q_data)):
                    dat = q_data[j]
                    q_dataArray[j, :len(dat)] = dat
                u2q_seq[user_id] = q_dataArray

                qa_dataArray = np.zeros((len(qa_data), self.seqlen))
                for j in range(len(qa_data)):
                    dat = qa_data[j]
                    qa_dataArray[j, :len(dat)] = dat
                u2qa_seq[user_id] = qa_dataArray

                tf_dataArray = np.zeros((len(tf_data), self.seqlen))
                for j in range(len(tf_data)):
                    dat = tf_data[j]
                    tf_dataArray[j, :len(dat)] = dat
                u2tf_seq[user_id] = tf_dataArray

                q_data.clear()
                qa_data.clear()
                tf_data.clear()
        f_data.close()
        return u2q_seq, u2qa_seq, u2tf_seq