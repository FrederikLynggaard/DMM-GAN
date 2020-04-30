from __future__ import print_function

import errno

from config import cfg, cfg_from_file
from inception.inception_score import inception_score

from importlib import import_module
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def evaluate(netG, image_encoder, text_encoder, dataset, dataloader, output_dir, name):
    cnt = 0
    R_count = 0
    R = np.zeros(30000)
    cont = True
    ii = 0
    while cont:
        for step, data in enumerate(dataloader, 0):
            cnt += dataloader.batch_size

            if cnt % 1000 == 0:
                print('cnt: ', cnt)

            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            hidden = text_encoder.init_hidden(dataloader.batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            #######################################################
            # (2) Generate fake images
            ######################################################
            nz = 100 # TODO refactor
            noise = Variable(torch.FloatTensor(dataloader.batch_size, nz), volatile=True)
            noise = noise.cuda()
            noise.data.normal_(0, 1)
            fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
            for j in range(dataloader.batch_size):
                s_tmp = '%s/%s/%s' % (output_dir, name, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    # print('Make a new folder: ', folder)
                    os.chdir(main_wd)
                    mkdir_p(folder)
                    os.chdir(model_wd)
                im = fake_imgs[-1][j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%d.png' % (s_tmp, ii)
                os.chdir(main_wd)
                im.save(fullpath)
                os.chdir(model_wd)

            _, cnn_code = image_encoder(fake_imgs[-1])

            for i in range(dataloader.batch_size):
                mis_captions, mis_captions_len = dataset.get_mis_caption(class_ids[i])
                hidden = text_encoder.init_hidden(99)
                _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                ### cnn_code = 1 * nef
                ### rnn_code = 100 * nef
                scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                scores0 = scores / norm.clamp(min=1e-8)
                if torch.argmax(scores0) == 0:
                    R[R_count] = 1
                R_count += 1

            if R_count >= 100:
                sum = np.zeros(10)
                np.random.shuffle(R)
                for i in range(10):
                    sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                R_mean = np.average(sum)
                R_std = np.std(sum)
                print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                cont = False
                break
    ii += 1

    # calculate IS
    os.chdir(main_wd)
    inception_score(output_dir+'/'+name)
    os.chdir(model_wd)




if __name__ == "__main__":
    main_wd = os.getcwd()
    cfg_from_file('experiments_bird.yml')

    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, cfg.RUN, timestamp)
    mkdir_p(output_dir)

    os.chdir(main_wd)
    model_info = cfg.MODELS[cfg.RUN]
    model_wd = model_info.WORKING_DIR
    os.chdir(model_wd)

    config_py = import_module('miscc.config')
    cfg_from_file_x = getattr(config_py, 'cfg_from_file')
    cfg_x = getattr(config_py, 'cfg')
    cfg_from_file_x(model_info.CONFIG_PATH)

    model_py = import_module('model')
    RNN_ENCODER = getattr(model_py, 'RNN_ENCODER')
    CNN_ENCODER = getattr(model_py, 'CNN_ENCODER_MOCK')
    G_NET = getattr(model_py, 'G_NET')

    utils_py = import_module('miscc.utils')
    weights_init = getattr(utils_py, 'weights_init')

    datasets_py = import_module('datasets')
    TextDataset = getattr(datasets_py, 'TextDataset')
    prepare_data = getattr(datasets_py, 'prepare_data')

    for version in model_info.VERSIONS.keys():
        version_info = model_info.VERSIONS[version]

        manual_seed = 100
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(manual_seed)

        # Get data loader
        split_dir, bshuffle = 'test', True
        imsize = cfg_x.TREE.BASE_SIZE * (2 ** (cfg_x.TREE.BRANCH_NUM - 1))
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        dataset = TextDataset(cfg_x.DATA_DIR, split_dir,
                              base_size=cfg_x.TREE.BASE_SIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg_x.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))



        # load image encoder
        image_encoder = CNN_ENCODER(cfg_x.TEXT.EMBEDDING_DIM)
        #state_dict = torch.load(model_info.IMAGE_ENCODER_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
        #image_encoder.load_state_dict(state_dict)
        print('Load image encoder from:', model_info.IMAGE_ENCODER_WEIGHTS_PATH)
        image_encoder = image_encoder.cuda()
        image_encoder.eval()

        # load generator network
        if 'CONFIG_PATH' in version_info.keys():
            cfg_from_file_x(version_info.CONFIG_PATH)
        netG = G_NET()
        netG.apply(weights_init)
        netG.cuda()
        netG.eval()
        model_dir = version_info.G_NET_WEIGHTS_PATH
        state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', model_dir)

        # load text encoder
        text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg_x.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(model_info.TEXT_ENCODER_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', model_info.TEXT_ENCODER_WEIGHTS_PATH)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        start_t = time.time()
        evaluate(netG, image_encoder, text_encoder, dataset, dataloader, output_dir, '{}_{}'.format(cfg.RUN, version))
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
