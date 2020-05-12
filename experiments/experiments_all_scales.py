from __future__ import print_function

import errno

from config import cfg, cfg_from_file

from importlib import import_module
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
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


def sampling():
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
            if cfg.RUN == 'DMM-GAN' or cfg.RUN == 'MirrorGAN':
                mask = (captions == 0) + (captions == 1) + (captions == 2)  # masked <start>, <end>, <pad>
            elif cfg.RUN == 'AttnGAN' or cfg.RUN == 'DM-GAN':
                mask = (captions == 0)
            else:
                raise AttributeError

            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            #######################################################
            # (2) Generate fake images
            ######################################################
            nz = cfg_x.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(dataloader.batch_size, nz), volatile=True)
            noise = noise.cuda()
            noise.data.normal_(0, 1)

            fake_imgs = []
            att_maps = []
            c_code, mu, logvar = netG.ca_net(sent_emb)

            if cfg_x.TREE.BRANCH_NUM > 0:
                h_code1 = netG.h_net1(noise, c_code)
                fake_img1 = netG.img_net1(h_code1)
                fake_imgs.append(fake_img1)
            if cfg_x.TREE.BRANCH_NUM > 1:
                h_code2, att1 = netG.h_net2(h_code1, c_code, words_embs, mask)
                fake_img2 = netG.img_net2(h_code2)
                fake_imgs.append(fake_img2)
                if att1 is not None:
                    att_maps.append(att1)
            if cfg_x.TREE.BRANCH_NUM > 2:
                h_code3, att2 = netG.h_net3(h_code2, c_code, words_embs, mask)
                fake_img3 = netG.img_net3(h_code3)
                fake_imgs.append(fake_img3)
                if att2 is not None:
                    att_maps.append(att2)

            for j in range(dataloader.batch_size):
                s_tmp = '%s/%s/images/epoch_%s/%s' % (output_dir, name, epoch, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    # print('Make a new folder: ', folder)
                    os.chdir(main_wd)
                    mkdir_p(folder)
                    os.chdir(model_wd)

                ca_path = '%s_%d_ca.pt' % (s_tmp, ii)
                torch.save(c_code[j], ca_path)
                noise_path = '%s_%d_noise.pt' % (s_tmp, ii)
                torch.save(noise[j], noise_path)

                for k in range(3):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_%d_%d.png' % (s_tmp, ii, 64*2**k)
                    os.chdir(main_wd)
                    im.save(fullpath)
                    os.chdir(model_wd)

            for i in range(dataloader.batch_size):
                R_count += 1

            if R_count >= 2000:
                cont = False
                break
        ii += 1


if __name__ == "__main__":
    main_wd = os.getcwd()
    cfg_from_file('experiments_bird.yml')

    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/%s_%s_%s_%s' % \
                 (cfg.OUTPUT_PATH, cfg.DATASET_NAME, cfg.CONFIG_NAME, cfg.RUN, timestamp)

    output_global_dir = '%s/global' % cfg.OUTPUT_PATH
    mkdir_p(output_dir)
    mkdir_p(output_global_dir)

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
    CNN_ENCODER = getattr(model_py, 'CNN_ENCODER')
    G_NET = getattr(model_py, 'G_NET')

    utils_py = import_module('miscc.utils')
    weights_init = getattr(utils_py, 'weights_init')

    datasets_py = import_module('datasets')
    TextDataset = getattr(datasets_py, 'TextDataset')
    prepare_data = getattr(datasets_py, 'prepare_data')

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

    # load text encoder
    text_encoder_path = os.path.join(cfg.MODELS_BASE_PATH, model_info.TEXT_ENCODER_WEIGHTS_PATH)
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg_x.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    print('Load text encoder from:', text_encoder_path)
    text_encoder = text_encoder.cuda()
    text_encoder.eval()

    results = {}

    for version in model_info.VERSIONS.keys():
        version_info = model_info.VERSIONS[version]
        name = '{}_{}'.format(cfg.RUN, version)

        results[name] = {}

        manual_seed = 100
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(manual_seed)

        # 0, 50, ..., 600
        for epoch in range(0, 601, 50):

            results[name][epoch] = {}

            print('--- Sampling all scales --- {}_{} ---'.format(cfg.RUN, version))

            # load generator network
            if 'CONFIG_PATH' in version_info.keys():
                cfg_from_file_x(version_info.CONFIG_PATH)
            netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            model_dir = os.path.join(cfg.MODELS_BASE_PATH, version_info.G_NET_WEIGHTS_PATH, 'netG_epoch_{}.pth'.format(epoch))
            print('Load G from: ', model_dir)
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('G loaded')

            start_t = time.time()
            sampling()
            end_t = time.time()

            print('Time:', end_t - start_t)