# -*- coding: utf-8 -*-
import json
from six.moves import cPickle
import h5py
import os
import numpy as np
import random
from multiprocessing import Pool
# from multiprocessing.dummy import Pool
import opts
# from joblib import Parallel, delayed
import time


def get_npy_data(image_id, fc_file, att_file):
    return (np.load(fc_file),
            np.load(att_file)['feat'],
            image_id)


def get_npy_feat_array(image_id, fc_file_list, att_file_list):
    fc_feat = []
    att_feat = []
    num_feat_array = len(fc_file_list)
    for i in range(num_feat_array):
        fc_feat.append(np.load(fc_file_list[i]))
        att_feat.append(np.load(att_file_list[i])['feat'])

    return fc_feat, att_feat, image_id


class DataLoader:

    def reset_iterator(self, split):
        self._prefetch_process[split].terminate()
        self._prefetch_process[split].join()
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt

        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.online_training = opt.online_training

        self.use_flip = opt.use_flip
        self.use_crop = opt.use_crop
        self.top_words_path = opt.top_words_path
        self.top_words_count = opt.top_words_count
        self.use_official_split = opt.use_official_split
        self.official_train_id_file = opt.official_train_id_file
        self.official_val_id_file = opt.official_val_id_file
        self.official_test_id_file = opt.official_test_id_file
        self.feature_type = opt.feature_type
        self.caption_model = opt.caption_model
        self.aug_type = opt.aug_type

        self.feat_array_info = opt.feat_array_info
        self.num_feat_array = len(self.feat_array_info)

        self.fc_feat_file_list = []
        self.att_feat_file_list = []
        self.feat_array_crops = ['original', 'flip', 'crop_tr', 'crop_tl', 'crop_bl', 'crop_br',
                                 'flip_crop_tr', 'flip_crop_tl', 'flip_crop_bl', 'flip_crop_br']

        if self.feature_type == 'feat_array':
            for i in range(len(self.feat_array_crops)):
                k = self.feat_array_crops[i]
                fc_feat_file_list_temp = []
                att_feat_file_list_temp = []
                for j in range(self.num_feat_array):
                    fc_feat_file_list_temp.append(self.feat_array_info[j][k]['fc'])
                    att_feat_file_list_temp.append(self.feat_array_info[j][k]['att'])

                self.fc_feat_file_list.append(fc_feat_file_list_temp)
                self.att_feat_file_list.append(att_feat_file_list_temp)

            assert (self.num_feat_array == len(self.fc_feat_file_list[0]))

        else:
            self.input_fc_dir = opt.input_fc_dir
            self.input_fc_flip_dir = opt.input_fc_flip_dir
            self.input_fc_crop_dir = opt.input_fc_crop_dir
            self.input_fc_crop_tl_dir = opt.input_fc_crop_tl_dir
            self.input_fc_crop_bl_dir = opt.input_fc_crop_bl_dir
            self.input_fc_crop_br_dir = opt.input_fc_crop_br_dir
            self.input_fc_flip_crop_dir = opt.input_fc_flip_crop_dir
            self.input_fc_flip_crop_tl_dir = opt.input_fc_flip_crop_tl_dir
            self.input_fc_flip_crop_bl_dir = opt.input_fc_flip_crop_bl_dir
            self.input_fc_flip_crop_br_dir = opt.input_fc_flip_crop_br_dir

            self.input_att_dir = opt.input_att_dir
            self.input_att_flip_dir = opt.input_att_flip_dir
            self.input_att_crop_dir = opt.input_att_crop_dir
            self.input_att_crop_tl_dir = opt.input_att_crop_tl_dir
            self.input_att_crop_bl_dir = opt.input_att_crop_bl_dir
            self.input_att_crop_br_dir = opt.input_att_crop_br_dir
            self.input_att_flip_crop_dir = opt.input_att_flip_crop_dir
            self.input_att_flip_crop_tl_dir = opt.input_att_flip_crop_tl_dir
            self.input_att_flip_crop_bl_dir = opt.input_att_flip_crop_bl_dir
            self.input_att_flip_crop_br_dir = opt.input_att_flip_crop_br_dir

        self.image_id_to_index = {}

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        print(self.top_words_path)
        self.top_words = cPickle.load(open(self.top_words_path, 'rb'))['words']
        self.word_to_top_ix = {}  # start from 0

        for i in range(len(self.top_words)):
            self.word_to_top_ix[self.top_words[i]] = i

        # compute word to index
        self.word_to_ix = {}
        for k, v in self.ix_to_word.items():
            self.word_to_ix[v] = k

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape  # (616767, 16)
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]  # 123287
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]  # 123287

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        print('len images: ')
        print(len(self.info['images']))
        print(self.info['images'][0])
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            image_id = img['id']
            assert image_id not in self.image_id_to_index
            self.image_id_to_index[image_id] = ix
        print(len(self.image_id_to_index))

        # separate out indexes for each of the provided splits
        self.split_image_id = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):  # ix is index
            img = self.info['images'][ix]
            image_id = img['id']
            if img['split'] == 'train':
                self.split_image_id['train'].append(image_id)
            elif img['split'] == 'val':
                self.split_image_id['val'].append(image_id)
            elif img['split'] == 'test':
                self.split_image_id['test'].append(image_id)
            elif opt.train_only == 0:  # restval
                self.split_image_id['train'].append(image_id)

        if self.online_training:
            self.split_image_id['train'] = self.split_image_id['train'] + self.split_image_id['test']
            # self.split_ix['val'] = self.split_ix['test']

        if self.use_official_split:
            self.split_image_id = {'train': [], 'val': [], 'test': []}

            # train_split_file = open(self.official_split_path + '/official_train_id.txt')
            train_split_file = open(self.official_train_id_file)
            for line in train_split_file:
                ix = int(line.strip())
                self.split_image_id['train'].append(ix)
            train_split_file.close()

            # val_split_file = open(self.official_split_path + '/official_val_id.txt')
            val_split_file = open(self.official_val_id_file)
            for line in val_split_file:
                ix = int(line.strip())
                self.split_image_id['val'].append(ix)
            val_split_file.close()

            # test_split_file = open(self.official_split_path + '/official_test_id.txt')
            test_split_file = open(self.official_test_id_file)
            for line in test_split_file:
                ix = int(line.strip())
                self.split_image_id['test'].append(ix)
            test_split_file.close()

        print('assigned %d images to split train' % len(self.split_image_id['train']))
        print('assigned %d images to split val' % len(self.split_image_id['val']))
        print('assigned %d images to split test' % len(self.split_image_id['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                self._prefetch_process[split].terminate()
                self._prefetch_process[split].join()

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        if self.feature_type == 'feat_array':
            fc_batch = [[] for _ in range(self.num_feat_array)]
            att_batch = [[] for _ in range(self.num_feat_array)]
        else:
            fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
            att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            import time
            t_start = time.time()
            # fetch image, tmp_att is numpy ndarray
            if self.feature_type == 'feat_array':
                tmp_fc_feat_array, tmp_att_feat_array, tmp_image_id, tmp_wrapped = \
                    self._prefetch_process[split].get()
                for feat_id in range(self.num_feat_array):
                    tmp_att = tmp_att_feat_array[feat_id]
                    if len(tmp_att.shape) == 3:
                        tmp_att = tmp_att.reshape(-1, tmp_att.shape[2])
                    att_batch[feat_id] += [tmp_att] * seq_per_img
                    fc_batch[feat_id] += [tmp_fc_feat_array[feat_id]] * seq_per_img

            else:
                tmp_fc, tmp_att, tmp_image_id, tmp_wrapped = self._prefetch_process[
                    split].get()
                if len(tmp_att.shape) == 3:
                    tmp_att = tmp_att.reshape(-1, tmp_att.shape[2])
                fc_batch += [tmp_fc] * seq_per_img
                att_batch += [tmp_att] * seq_per_img

            if tmp_image_id in self.image_id_to_index:
                ix = self.image_id_to_index[tmp_image_id]
            else:
                ix = -1

            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            if ix >= 0:
                # fetch the sequence labels
                ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
                ix2 = self.label_end_ix[ix] - 1
                ncap = ix2 - ix1 + 1  # number of captions available for this image
                assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

                if ncap < seq_per_img:
                    # we need to subsample (with replacement)
                    seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                    for q in range(seq_per_img):
                        ixl = random.randint(ix1, ix2)
                        seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                else:
                    ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                    seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1:self.seq_length + 1] = seq

            # Used for reward evaluation
            if ix >= 0:
                gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append(seq)

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            if ix >= 0:
                info_dict['id'] = self.info['images'][ix]['id']
            else:
                info_dict['id'] = tmp_image_id

            assert tmp_image_id == info_dict['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            # print(i, time.time() - t_start)

            if tmp_wrapped:
                wrapped = True

        # generate mask
        t_start = time.time()
        # nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        nonzeros = np.sum(label_batch != 0, axis=1) + 2
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        # print('mask', time.time() - t_start)

        # generate top 1000 words
        top_1000 = np.zeros((label_batch.shape[0], self.top_words_count), dtype=np.int)
        top_1000.fill(-1)

        for i in range(label_batch.shape[0]):
            top_temp = {}
            for j in range(label_batch.shape[1]):
                w_index = label_batch[i][j]
                if not w_index == 0:
                    word = self.ix_to_word[str(w_index)]
                    if word in self.word_to_top_ix:
                        top_temp[self.word_to_top_ix[word]] = 1

            top_list = list(top_temp.keys())
            for k in range(len(top_list)):
                top_1000[i][k] = top_list[k]

        data = {}
        if self.feature_type == 'feat_array':
            fc_data_all = []
            att_data_all = []
            for feat_id in range(self.num_feat_array):
                fc_data_all.append(np.stack(fc_batch[feat_id]))
                att_data_all.append(np.stack(att_batch[feat_id]))

            data['fc_feats_array'] = fc_data_all
            data['att_feats_array'] = att_data_all
        else:
            data['fc_feats'] = np.stack(fc_batch)
            data['att_feats'] = np.stack(att_batch)

        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_image_id[split]),
                          'wrapped': wrapped}
        data['infos'] = infos
        data['top_words'] = top_1000

        return data

    # print info of a batch
    def print_batch(self, data):
        if self.feature_type == 'feat_array':
            for feat_id in range(self.num_feat_array):
                print(data['fc_feats_array'][feat_id].shape)
                print(data['att_feats_array'][feat_id].shape)

        else:
            print(data['fc_feats'].shape)
            print(data['att_feats'].shape)

        info = data['infos']
        label_batch = data['labels']
        top_words = data['top_words']
        mask = data['masks']
        batch_size, t = label_batch.shape

        cap_per_image = batch_size / len(info)
        for i in range(batch_size):
            image_id = info[int(i / cap_per_image)]['id']
            cap = ''
            mask_str = ''
            index_str = ''
            for j in range(t):
                word_index = label_batch[i][j]
                index_str = index_str + str(word_index) + ' '
                mask_str = mask_str + str(mask[i][j]) + ' '
                if not word_index == 0:
                    cap = cap + self.ix_to_word[str(word_index)] + ' '

            top = ''
            for a in range(self.top_words_count):
                if top_words[i][a] >= 0:
                    top_word_index = top_words[i][a]
                    top = top + self.top_words[top_word_index] + ' '


class BlobFetcher:
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

        self.pool = Pool(8)
        # self.read_file_pool = Pool(4)
        self.fifo = []
        self.cur_idx = self.dataloader.iterators[self.split]  # index for loading features
        self.cur_split_image_id = self.dataloader.split_image_id[self.split][:]  # copy
        self.feature_type = dataloader.feature_type

        self.feat_array_info = dataloader.feat_array_info
        self.num_feat_array = dataloader.num_feat_array

    # Add more in the queue
    def reset(self):
        if len(self.fifo) == 0:
            self.cur_idx = self.dataloader.iterators[self.split]
            self.cur_split_image_id = self.dataloader.split_image_id[self.split][:]  # copy

        for i in range(512 - len(self.fifo)):
            image_id = self.cur_split_image_id[self.cur_idx]

            if self.cur_idx + 1 >= len(self.cur_split_image_id):
                self.cur_idx = 0
                if self.if_shuffle:
                    random.shuffle(self.cur_split_image_id)
            else:
                self.cur_idx += 1

            if self.feature_type == 'feat_array':
                if self.dataloader.use_flip:  # use flip and crop
                    if self.dataloader.use_crop:
                        flip_type = np.random.randint(0, 10)
                    else:
                        flip_type = np.random.randint(0, 2)
                else:
                    flip_type = self.dataloader.aug_type
                # print('---- flip_type: ' + str(flip_type))
                fc_feat_file_list = self.dataloader.fc_feat_file_list[flip_type]
                att_feat_file_list = self.dataloader.att_feat_file_list[flip_type]
                fc_feat_file_list_com = []
                att_feat_file_list_com = []
                for a in fc_feat_file_list:
                    fc_feat_file_list_com.append(os.path.join(a, str(image_id) + '.npy'))
                for a in att_feat_file_list:
                    att_feat_file_list_com.append(os.path.join(a, str(image_id) + '.npz'))

                self.fifo.append(self.pool.apply_async(get_npy_feat_array,
                                                       (image_id,
                                                        fc_feat_file_list_com,
                                                        att_feat_file_list_com
                                                        )))

            else:  # other feature type
                if self.dataloader.use_flip:  # use aug
                    if self.dataloader.use_crop:
                        flip_type = np.random.randint(0, 10)
                    else:
                        flip_type = np.random.randint(0, 2)

                    if flip_type == 0:
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 1:  # flip
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 2:  # crop top right
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_crop_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_crop_dir,
                                                                             str(image_id) + '.npz')
                                                                )))

                    elif flip_type == 3:  # flip and crop top right
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_crop_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_crop_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 4:  # crop top left
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_crop_tl_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_crop_tl_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 5:  # flip and crop top left
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_crop_tl_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_crop_tl_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 6:  # crop bottom left
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_crop_bl_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_crop_bl_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 7:  # flip and crop bottom left
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_crop_bl_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_crop_bl_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 8:  # crop bottom right
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_crop_br_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_crop_br_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif flip_type == 9:  # flip and crop bottom right
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_crop_br_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_crop_br_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    else:
                        raise Exception("flip_type not supported: {}".format(flip_type))

                else:
                    if self.dataloader.aug_type == 0:  # origin feature
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif self.dataloader.aug_type == 1:
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif self.dataloader.aug_type == 2:
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_crop_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_crop_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    elif self.dataloader.aug_type == 3:
                        self.fifo.append(self.pool.apply_async(get_npy_data,
                                                               (image_id,
                                                                os.path.join(self.dataloader.input_fc_flip_crop_dir,
                                                                             str(image_id) + '.npy'),
                                                                os.path.join(self.dataloader.input_att_flip_crop_dir,
                                                                             str(image_id) + '.npz')
                                                                )))
                    else:
                        raise Exception("aug_type not supported: {}".format(self.dataloader.aug_type))

    def terminate(self):
        while len(self.fifo) > 0:
            self.fifo.pop(0).get()
        self.pool.terminate()
        # self.read_file_pool.terminate()
        print(self.split, 'terminated')

    def join(self):
        self.pool.join()
        # self.read_file_pool.join()
        print(self.split, 'joined')

    def _get_next_minibatch_inds(self):
        max_index = len(self.cur_split_image_id)
        wrapped = False

        ri = self.dataloader.iterators[self.split]  # index for reading features.
        image_id = self.dataloader.split_image_id[self.split][ri]
        # ix = self.dataloader.split_image_id[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            self.dataloader.split_image_id[self.split] = self.cur_split_image_id[:]  # copy
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return image_id, wrapped

    def get(self):
        if len(self.fifo) < 100:
            self.reset()

        image_id, wrapped = self._get_next_minibatch_inds()
        tmp = self.fifo.pop(0).get()

        if self.feature_type == 'feat_array':
            assert tmp[2] == image_id, "image_id not equal"
        else:
            assert tmp[2] == image_id, "image_id not equal"

        return tmp + (wrapped,)


if __name__ == '__main__':
    opt = opts.parse_opt()
    loader = DataLoader(opt)

    start = time.time()
    for n in range(100):
        print('---- batch: ' + str(n))
        data = loader.get_batch('test')

    start2 = time.time()
    print('time: ', start2 - start)
