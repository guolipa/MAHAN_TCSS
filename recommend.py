import os
import math
import random
import pickle
import tqdm
# import scipy.sparse as sparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR

from models_dist import Train_Model
from data_precess_dist import DataSet
from evaluate import *

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T

def get_candidatelist(epoch, test_dist, dataset, model, config, R, flag='test'):
    # R = config['R']
    test_geo = defaultdict()
    location = dataset.location
    candidatelist = defaultdict(dict) 
    all_venues = range(dataset.item_nums)  
    if flag == 'test':
        test = dataset.test
        user = test['test_user']
    else:
        test = dataset.valid
        user = test['valid_user']

    for i, uid in enumerate(tqdm.tqdm(user, desc="test")):
        train_checks = dataset.data[uid]['item'][:-1]  
        if flag == 'test':
            target = test['test_target_item'][i]
            target_time = test['test_target_time'][i]
            history = test['test_history'][i]
            seq = test['test_seq_item'][i]
            seq_time = test['test_seq_time'][i]
            seq_dist = test['test_seq_dist'][i]
            # seq_dist = [1]
            delatime = test['test_delatime'][i]
        else:
            target = test['valid_target_item'][i]
            history = test['valid_history'][i]
            seq = test['valid_seq_item'][i]
        if target in train_checks:
            continue
        if config['recommend_new']:
            recommend_list = np.setdiff1d(np.array(all_venues), np.array(train_checks))  

            current_location = list(location.loc[location.vid == target].values[0])[1:]  
            x, y = current_location
            lat_max = R / 111 + x
            lat_min = x - R / 111
            lon_max = R / (111 * np.cos(x * math.pi / 180.0)) + y
            lon_min = y - R / (111 * np.cos(x * math.pi / 180.0))
            near_location = location[(location["lon"] > lon_min) & \
                                    (location["lon"] < lon_max) & \
                                    (location["lat"] > lat_min) & \
                                    (location["lat"] < lat_max)]
            neighbors = list(np.intersect1d(near_location.vid.values, recommend_list))
            # geo_dist = []
            if epoch == 0:
                #geo_dist = dataset.place_correlation[neighbors][:, history].toarray()
                geo_dist = []
                # test_geo[uid] = geo_dist
            else:
                # geo_dist = test_geo[uid]
                geo_dist = []
            overall_scores = model(T.LongTensor([uid] * len(neighbors)), T.LongTensor(seq), T.LongTensor(seq_time), T.LongTensor(history), seq_dist, geo_dist,
                                   T.LongTensor(neighbors), T.LongTensor([target_time] * len(neighbors)), T.LongTensor(delatime), flag='test').cpu().detach().numpy()
            predict_scores = zip(neighbors, list(overall_scores))
            predict_scores = sorted(predict_scores, key=lambda x: x[1], reverse=True)[0:100]
            candidatelist[uid][target] = [x[0] for x in predict_scores]
        else:
            current_location = list(location.loc[location.vid == target].values[0])[1:] 
            x, y = current_location
            lat_max = R / 111 + x
            lat_min = x - R / 111
            lon_max = R / (111 * np.cos(x * math.pi / 180.0)) + y
            lon_min = y - R / (111 * np.cos(x * math.pi / 180.0))
            near_location = location[(location["lon"] > lon_min) & \
                                     (location["lon"] < lon_max) & \
                                     (location["lat"] > lat_min) & \
                                     (location["lat"] < lat_max)]
            neighbors = list(near_location.vid.values)
            overall_scores = model(T.LongTensor([uid]*len(neighbors)), T.LongTensor(seq), T.LongTensor(history), T.LongTensor(neighbors),
                                   flag='test').cpu().detach().numpy()
            predict_scores = zip(neighbors, list(overall_scores))
            predict_scores = sorted(predict_scores, key=lambda x: x[1], reverse=True)[0:100]
            candidatelist[uid][target] = [x[0] for x in predict_scores]
    return candidatelist, test_geo

def main(dataset, config, model_file, f):
    loss = []
    test_dist = defaultdict()
    if config['train']:
        train_model = Train_Model(config)
        if config['pre_train']:
            print('*' * 15 + 'Start Pretraining ' + '*' * 15)
            config['model'] = 'model_long'
            train_model.model.model = 'model_long'
            print(train_model.model.model)
            pre_loss = []
            for epoch in range(10):
                # train_model.model.train()  
                uids = list(range(config['user_nums']))
                random.shuffle(uids)
                total_loss = 0
                batch_num = 0
                for batch_id, uid in enumerate(tqdm.tqdm(uids, desc="pre-train")):
                    batch = dataset.get_train_batch_user(uid)
                    if torch.cuda.is_available():
                        users, seq_items, seq_times, hist_items, target_items, target_times, labels, s_mask, h_mask = \
                        batch[0].cuda(), \
                        batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), \
                        batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda()
                    else:
                        users, seq_items, seq_times, hist_items, target_items, target_times, labels, s_mask, h_mask = \
                        batch[0], batch[1], \
                        batch[2], batch[3], batch[4], \
                        batch[5], batch[6], batch[7], batch[8]
                    batch_loss = train_model.train_batch(users, seq_items, seq_times, hist_items, target_items,
                                                         target_times, labels, s_mask, h_mask)
                    total_loss += batch_loss
                    batch_num += 1

                total_loss = total_loss / batch_num
                print('Epoch {},loss {}'.format(epoch, total_loss))
                pre_loss.append(total_loss)

                candidatelist = get_candidatelist(epoch, dataset, train_model.model, config, config['R'])
                recall = []
                ndgc = []
                for N in range(5, 55, 5):
                    r = getHitRation(candidatelist, N)
                    n = getNDCG(candidatelist, N)
                    recall.append(r)
                    ndgc.append(n)
                print(recall)
                print(ndgc)
            print(pre_loss)
            # train_model.save_model(pre_train_model_file)
            print('*' * 15 + 'Finish Pretraining ' + '*' * 15)
            config['model'] = 'model_fuse'
            train_model.model.model = 'model_fuse'

        print('*' * 15 + 'Start training model' + '*' * 15)
        print(train_model.model.model)

        # scheduler = StepLR(optimizer=train_model.optimizer, step_size=3, gamma=0.6)
        # scheduler = MultiStepLR(optimizer=train_model.optimizer, milestones=range(5, 15, 5), gamma=0.5)
        scheduler = ExponentialLR(optimizer=train_model.optimizer, gamma=0.9)

        for epoch in range(config['num_epochs']):
            print('Start training epoch {}'.format(epoch))
            print('-' * 30)

            train_model.model.train()  # 打开dropout和layer norm

            uids = list(range(config['user_nums']))
            random.shuffle(uids)

            total_loss = 0
            batch_num = 0
            for batch_id, uid in enumerate(tqdm.tqdm(uids, desc="train")):
                batch = dataset.get_train_batch_user(uid)
                if torch.cuda.is_available():
                    users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes,labels, s_mask, h_mask = batch[0].cuda(), \
                                                    batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4],\
                                                    batch[5], batch[6].cuda(), batch[7].cuda(), batch[8].cuda(), \
                                                    batch[9].cuda(), batch[10].cuda(), batch[11].cuda()
                else:
                    users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes, labels, s_mask, h_mask = batch[0], batch[1], \
                                                                                        batch[2], batch[3], batch[4], \
                                                                                        batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11]
                batch_loss = train_model.train_batch(users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes, labels, s_mask, h_mask)
                # print('Training epoch {},batch {},loss {}'.format(epoch, batch_id, batch_loss))
                total_loss += batch_loss
                batch_num += 1

            total_loss = total_loss / batch_num
            print('Epoch {},loss {}'.format(epoch, total_loss))
            loss.append(total_loss)

            torch.cuda.empty_cache()
            ev = 1
            if config['seq_len'] == 6:
                ev = 1
            if (epoch+1) % ev == 0:
                train_model.model.eval()   

                with torch.no_grad():
                    candidatelist, test_d = get_candidatelist(epoch, test_dist, dataset, train_model.model, config, config['R'])
                    # with open('/home1/zcw/guolipa/CAMN/candidate_f_short.pk', 'wb') as f:
                    # pickle.dump(candidatelist, f)
                    if epoch == 0:
                        test_dist = test_d

                recall = []
                ndgc = []
                mrp = []
                for N in range(5, 55, 5):
                    r = getHitRation(candidatelist, N)
                    n = getNDCG(candidatelist, N)
                    m = getMRP(candidatelist, N)
                    recall.append(r)
                    ndgc.append(n)
                    mrp.append(m)

                print(recall[:4])
                print(ndgc[:4])

                f.write('epoch: ' + str(epoch) + '\n')
                f.write(':'.join(['recall', ','.join([str(r) for r in recall])]) + '\n')
                f.write(':'.join(['ndgc', ','.join([str(r) for r in ndgc])]) + '\n')
                f.write(':'.join(['mrp', ','.join([str(r) for r in mrp])]) + '\n')
                f.flush()

            scheduler.step()
            torch.cuda.empty_cache()
            print(train_model.model.Short_Model.attn.a)
        print(loss)
        f.write(':'.join(['loss', ','.join([str(r) for r in loss])]) + '\n')

        #train_model.save_model(model_file)
        print('--------------------Model training id finished-------------------------------------')
        print('--------------------Start recommending--------------------------------------------')
        # candidatelist = get_candidatelist(dataset, train_model.model, config['R'])
        return loss, candidatelist
    # else:
    #     model = Model_Long(config)
    #     model.load_state_dict(torch.load(model_file, map_location='cpu'))
    #     print('--------------------Start recommending--------------------------------------------')
    #     candidatelist = get_candidatelist(dataset, model, config['R'])
    #     print('--------------------Recommend is finished--------------------------------------------')
    #     return loss, candidatelist

def run(f, sn):
    config = {'user_nums': 23686,
              'item_nums': 27055,
              'emb_dim': 50,
              'key_dim': 50,
              'value_dim': 50,
              'hidden_dim': 50,
              'attention_dim': 50,
              'dim_feedforward': 50,
              'slot_num': 5,
              'hop_num': 1,
              'seq_len': 6,
              'max_len': 50,
              'max_relative_position': 5,
              'target_len': 1,
              'num_heads': 2,
              'alpha': 0.8,
              'neg_nums': 1,
              'pos_encod': False,
              'num_directions': 1,  # RNN
              'loss_fuction': 'BCELoss',  # 'BCELoss', 'BPRLoss
              'optimizer': 'adam',
              'adam_lr': 0.001,
              'l2_regularization': 0.000005,
              'dropout': 0.5,
              'num_epochs': 30,
              'batch_size': 128,
              'R': 100,
              'model': 'model_short',  # 'model_long', 'model_short','model_fuse', 'rnn', 'rnn_attention'
              'rnn_type': 'LSTM',  # 'RNN'  'GRU', 'LSTM'
              'long_method': 'feature',  # 'item', 'feature'
              'short_method': 'Self_Attention',  # 'Self_Attention', 'RNN'
              'sf_method': 'avg',  # 'user', 'location', 'time'
              'co_attention': True,
              'pre_train': False,
              'recommend_new': True,
              'train': True,
              'recommend': True,
              }
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config['seq_len'] = sn
    print(config['seq_len'])

    file = '/home1/zcw/guolipa/CAMN/'
    data_file = '/home1/zcw/guolipa/data/Foursquare/checkins.csv'
    model_file = file + 'STSA_f.pth'

    print('--------------------Processing data--------------------------------------------')
    data = pd.read_csv(data_file, sep=',', encoding='latin-1')
    dataset = DataSet(data, config)
    print('--------------------Processing data done--------------------------------------------')
    config['user_nums'] = dataset.user_nums
    config['item_nums'] = dataset.item_nums

    loss, candidatelist = main(dataset, config, model_file, f)

    print('done.....')
    return loss


if __name__ == '__main__':
    file = '/home1/zcw/guolipa/CAMN/'
    result_file = file + 'result_f_short_+S0.6+T.txt'
    f = open(result_file, 'w')

    f.write('===============seq_len===============' + '\n')
    for sl in [6]:
        f.write('---------------seq_len:' + str(sl) + '---------------' + '\n')
        loss = run(f, sl)

    f.close()


