import sys
import os
import logging
import argparse
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import run
import math
from load_data import DATA
from model import MODEL
from run import train, test


# root = logging.getLogger()
# root.setLevel(logging.DEBUG)
#
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# root.addHandler(ch)

def find_file(dir_name, best_epoch):
    for dir, subdir, files in os.walk(dir_name):
        for sub in subdir:
            if sub[0:len(best_epoch)] == best_epoch and sub[len(best_epoch)] == "_":
                return sub

def load_params(prefix, epoch):
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def train_one_dataset(params, file_name, train_q_data, train_qa_data, valid_q_data, valid_qa_data, valid_tf_data):
    ### ================================== model initialization ==================================
    g_model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    net = mx.mod.Module(symbol=g_model.sym_gen(),
                        data_names=['q_data', 'qa_data'],
                        label_names=['target'],
                        context=params.ctx, )
    # create memory by given input shapes
    net.bind(data_shapes=[mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
                          mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
             label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])

    # initial parameters with the default DKVMN initializer
    init_dkvmn_param_file_name = params.save + '-dkvmn_initialization'
    arg_params, aux_params = load_params(prefix=os.path.join('model', params.load, init_dkvmn_param_file_name), epoch=30)
    net.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
    '''
    # initial parameters with the default random initializer
    net.init_params(initializer=mx.init.Normal(sigma=params.init_std), force_init=True)    
    '''


    # decay learning rate in the lr_scheduler
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=20 * (train_q_data.shape[0] / params.batch_size),
                                                   factor=0.667, stop_factor_lr=1e-5)

    net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': params.lr, 'momentum': params.momentum,
                                                          'lr_scheduler': lr_scheduler})

    '''
    for parameters in net.get_params()[0]:
        print(parameters, net.get_params()[0][parameters].asnumpy().shape)
        #print(parameters, net.get_params()[0][parameters])
    print("\n")
    '''

    ### ================================== start training ==================================
    all_train_loss = {}
    all_train_acc = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_acc = {}
    all_valid_auc = {}

    best_valid_acc = -1
    best_valid_loss = -1

    for idx in range(params.max_iter):
        train_loss, train_acc = run.train(net, params, train_q_data, train_qa_data, label='Train')
        pred_list, target_list = run.test(net, params, valid_q_data, valid_qa_data, valid_tf_data, label='Valid')

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        valid_loss = run.binaryEntropy(all_target, all_pred)
        valid_acc = run.compute_accuracy(all_target, all_pred)

        '''
        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_acc\t", valid_acc, "\ttrain_acc\t", train_acc)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)
        '''

        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_acc[idx + 1] = valid_acc
        all_train_acc[idx + 1] = train_acc

        # output the epoch with the best validation auc
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss

            if not os.path.isdir('model'):
                os.makedirs('model')
            if not os.path.isdir(os.path.join('model', params.save)):
                os.makedirs(os.path.join('model', params.save))
            net.save_checkpoint(prefix=os.path.join('model', params.save, file_name), epoch=30)

    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        os.makedirs(os.path.join('result', params.save))
    f_save_log = open(os.path.join('result', params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_acc:\n" + str(all_valid_acc) + "\n\n")
    f_save_log.write("train_acc:\n" + str(all_train_acc) + "\n\n")
    f_save_log.close()

    return best_valid_acc, best_valid_loss


def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_tf_data, best_epoch, user_id):
    # print("\n\nStart testing ......................\n best_epoch:", best_epoch)
    g_model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    test_net = mx.mod.Module(symbol=g_model.sym_gen(),
                             data_names=['q_data', 'qa_data'],
                             label_names=['target'],
                             context=params.ctx)
    # create memory by given input shapes

    test_net.bind(data_shapes=[
        mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
        mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
        label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])

    arg_params, aux_params = load_params(prefix=os.path.join('model', params.load, file_name), epoch=best_epoch)
    test_net.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)

    pred_list, target_list = run.test(test_net, params, test_q_data, test_qa_data, test_tf_data, label='Test')

    return pred_list, target_list


def adjust_param(max_iter, batch_size, seqlen, test_seqlen, min_seqlen, max_seqlen):
    parser = argparse.ArgumentParser(description='Script to test KVMN.')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=max_iter, help='number of iterations')  # default=50
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')

    dataset = "STATICS"  # assist2009_updated / assist2015 / KDDal0506 / STATICS

    if dataset == "assist2009_updated":
        parser.add_argument('--batch_size', type=int, default=batch_size, help='the batch size')  # 32
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')  # 50
        parser.add_argument('--qa_embed_dim', type=int, default=10,
                            help='answer and question embedding dimensions')  # 200
        parser.add_argument('--memory_size', type=int, default=10, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=seqlen,
                            help='the allowed maximum length of a sequence')  # 200
        parser.add_argument('--data_dir', type=str, default='../../data/assist2009_updated', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
        parser.add_argument('--load', type=str, default='assist2009_updated', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')
    elif dataset == "assist2015":
        parser.add_argument('--batch_size', type=int, default=batch_size, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=10, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=10, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=seqlen, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/assist2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')
        parser.add_argument('--load', type=str, default='assist2015', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2015', help='path to save model')
    elif dataset == "STATICS":
        parser.add_argument('--batch_size', type=int, default=batch_size, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=10, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=1223,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=seqlen, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/STATICS', help='data directory')
        parser.add_argument('--data_name', type=str, default='STATICS', help='data set name')
        parser.add_argument('--load', type=str, default='STATICS', help='model file to load')
        parser.add_argument('--save', type=str, default='STATICS', help='path to save model')

    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    params.ctx = mx.cpu()

    # test_seqlen = params.seqlen
    #
    # Read data
    train_dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    test_dat = DATA(n_question=params.n_question, seqlen=test_seqlen, separate_char=',')
    seedNum = 224
    np.random.seed(seedNum)
    if not params.test:
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        train_seqlen = params.seqlen
        d = vars(params)

        train_data_path = params.data_dir + "/" + params.data_name + "_sub_train_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"
        valid_data_path = params.data_dir + "/" + params.data_name + "_sub_valid_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"
        test_data_path = params.data_dir + "/" + params.data_name + "_test_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"

        train_u2q_data, train_u2qa_data = train_dat.load_data(train_data_path)
        valid_u2q_data, valid_u2qa_data, valid_u2tf_data = train_dat.load_test_data(valid_data_path, 0.111)  # 0.1/0.9
        test_u2q_data, test_u2qa_data, test_u2tf_data = test_dat.load_test_data(test_data_path, 0.1)

        total_train_valid_acc = 0
        total_train_valid_loss = 0
        total_test_valid_auc = 0
        total_test_valid_acc = 0
        total_test_valid_loss = 0
        user_count = 0
        best_epoch = 30

        all_pred_list = []
        all_target_list = []
        i = 0
        for user_id in train_u2q_data:
            params.seqlen = train_seqlen
            file_name = 'u' + user_id + '_b' + str(params.batch_size) + \
                        '_q' + str(params.q_embed_dim) + '_qa' + str(params.qa_embed_dim) + \
                        '_m' + str(params.memory_size) + '_std' + str(params.init_std) + \
                        '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                        '_f' + str(params.final_fc_dim) + '_s' + str(seedNum)
            train_q_data = train_u2q_data[user_id]
            train_qa_data = train_u2qa_data[user_id]
            valid_q_data = valid_u2q_data[user_id]
            valid_qa_data = valid_u2qa_data[user_id]
            valid_tf_data = valid_u2tf_data[user_id]

            train_valid_acc, train_valid_loss = train_one_dataset(params, file_name, train_q_data, train_qa_data,
                                                                  valid_q_data, valid_qa_data, valid_tf_data)

            total_train_valid_acc += train_valid_acc
            total_train_valid_loss += train_valid_loss

            if params.train_test:
                params.seqlen = test_seqlen
                test_q_data = test_u2q_data[user_id]
                test_qa_data = test_u2qa_data[user_id]
                test_tf_data = test_u2tf_data[user_id]

                pred_list, target_list = test_one_dataset(params, file_name, test_q_data, test_qa_data, test_tf_data,
                                                          best_epoch, user_id)
                all_pred_list += pred_list
                all_target_list += target_list
            user_count += 1

        average_train_valid_acc = total_train_valid_acc / user_count
        average_train_valid_loss = total_train_valid_loss / user_count

        # print("average_train_valid_acc: ", average_train_valid_acc)
        # print("average_train_valid_loss: ", average_train_valid_loss)

        all_pred = np.concatenate(all_pred_list, axis=0)
        all_target = np.concatenate(all_target_list, axis=0)
        loss = run.binaryEntropy(all_target, all_pred)
        auc = run.compute_auc(all_target, all_pred)
        acc = run.compute_accuracy(all_target, all_pred)

        # print("valid_auc: ", auc)
        # print("valid_acc: ", acc)
        # print("valid_loss: ", loss)
        return auc
    else:
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        params.seqlen = test_seqlen
        d = vars(params)

        train_data_path = params.data_dir + "/" + params.data_name + "_sub_train_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"
        valid_data_path = params.data_dir + "/" + params.data_name + "_sub_valid_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"
        test_data_path = params.data_dir + "/" + params.data_name + "_test_"+str(min_seqlen)+"_"+str(max_seqlen)+".csv"

        train_u2q_data, train_u2qa_data = train_dat.load_data(train_data_path)
        test_u2q_data, test_u2qa_data, test_u2tf_data = test_dat.load_test_data(test_data_path, 0.1)

        user_count = 0
        best_epoch = 30

        all_pred_list = []
        all_target_list = []
        i = 0
        for user_id in train_u2q_data:
            file_name = params.save + '-dkvmn_initialization'

            test_q_data = test_u2q_data[user_id]
            test_qa_data = test_u2qa_data[user_id]
            test_tf_data = test_u2tf_data[user_id]

            pred_list, target_list = test_one_dataset(params, file_name, test_q_data, test_qa_data, test_tf_data,
                                                      best_epoch, user_id)
            all_pred_list += pred_list
            all_target_list += target_list

            user_count += 1

        all_pred = np.concatenate(all_pred_list, axis=0)
        all_target = np.concatenate(all_target_list, axis=0)
        loss = run.binaryEntropy(all_target, all_pred)
        auc = run.compute_auc(all_target, all_pred)
        acc = run.compute_accuracy(all_target, all_pred)

        # print("valid_auc: ", auc)
        # print("valid_acc: ", acc)
        # print("valid_loss: ", loss)

        return auc


if __name__ == '__main__':
    min_seqlen = 10
    max_seqlen = 20
    test_seqlen = 200
    max_iter = 1
    for i in range(9, 10):
        if i == 1:
            min_seqlen = 10
            max_seqlen = 20
            test_seqlen = 24
        elif i == 2:
            min_seqlen = 20
            max_seqlen = 40
            test_seqlen = 49
        elif i == 3:
            min_seqlen = 40
            max_seqlen = 80
            test_seqlen = 158
        elif i == 4:
            min_seqlen = 80
            max_seqlen = 120
            test_seqlen = 149
        elif i == 5:
            min_seqlen = 120
            max_seqlen = 160
            test_seqlen = 197
        elif i == 6:
            min_seqlen = 160
            max_seqlen = 200
            test_seqlen = 90
        elif i == 7:
            min_seqlen = 200
            max_seqlen = 300
            test_seqlen = 127
        elif i == 8:
            min_seqlen = 300
            max_seqlen = 500
            test_seqlen = 13
        elif i == 9:
            min_seqlen = 800
            max_seqlen = 900
            test_seqlen = 101

        max_auc = 0.0
        max_auc_batch_size = 0
        max_auc_seqlen = 0
        for k in range(20, 40):
            batch_size = k
            if min_seqlen / batch_size < 2:
                continue
            for j in range(2, int(min_seqlen / batch_size)):
                # for j in range(197, 198):
                seqlen = j
                auc = adjust_param(max_iter, batch_size, seqlen, test_seqlen, min_seqlen, max_seqlen)
                # print(j, auc)
                if auc > max_auc:
                    max_auc = auc
                    max_auc_batch_size = k
                    max_auc_seqlen = j
        print(i, "==>", max_iter, max_auc_batch_size, max_auc_seqlen, max_auc, '\n')