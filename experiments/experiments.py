from __future__ import print_function

import errno
import json

from config import cfg, cfg_from_file
from inception.inception_score import inception_score
from frechet_inception_distance.fid_score import compute_statistics_of_path
from frechet_inception_distance.fid_score import calculate_frechet_distance
from frechet_inception_distance.fid_score import InceptionV3

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


def sampling_and_r_precision():
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
            nz = cfg_x.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(dataloader.batch_size, nz), volatile=True)
            noise = noise.cuda()
            noise.data.normal_(0, 1)
            fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
            for j in range(dataloader.batch_size):
                s_tmp = '%s/%s/images/epoch_%s/%s' % (output_dir, name, epoch, keys[j])
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

            if R_count >= 30000:
                sum = np.zeros(10)
                np.random.shuffle(R)
                for i in range(10):
                    sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
                R_mean = np.average(sum)
                R_std = np.std(sum)
                results[name][epoch]['R'] = {'mean': '{:.5f}'.format(R_mean), 'std': '{:.5f}'.format(R_std)}
                print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                cont = False
                break
        ii += 1


def compute_is():
    fake_imgs_path = os.path.join(output_dir, name, 'images', 'epoch_{}'.format(epoch))

    mean, std = inception_score(fake_imgs_path)
    results[name][epoch]['IS'] = {'mean': '{:.5f}'.format(mean), 'std': '{:.5f}'.format(std)}
    print('IS: ', mean, std)


def compute_fid():
    fake_imgs_path = os.path.join(output_dir, name, 'images', 'epoch_{}'.format(epoch))
    dims = 2048

    global_fid_filename = os.path.join(output_global_dir, '{}_stats.npz'.format(cfg.DATASET_NAME))
    if os.path.exists(global_fid_filename):
        stats = np.load(global_fid_filename)
        m1 = stats['m1']
        s1 = stats['s1']
    else:
        m1, s1 = compute_statistics_of_path(cfg.DATASET_IMGS_PATH, fid_model, cfg.FID_BATCH_SIZE, dims, cfg.CUDA)
        np.savez(global_fid_filename, m1=m1, s1=s1)

    m2, s2 = compute_statistics_of_path(fake_imgs_path, fid_model, cfg.FID_BATCH_SIZE, dims, cfg.CUDA)
    fid = calculate_frechet_distance(m1, s1, m2, s2)
    results[name][epoch]['FID'] = '{:.5f}'.format(fid)
    print('FID: ', fid)


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

    # load image encoder
    image_encoder_path = os.path.join(cfg.MODELS_BASE_PATH, model_info.TEXT_ENCODER_WEIGHTS_PATH).replace(
        'text_encoder', 'image_encoder')
    image_encoder = CNN_ENCODER(cfg_x.TEXT.EMBEDDING_DIM)
    print('Load image encoder from:', image_encoder_path)
    state_dict = torch.load(image_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder = image_encoder.cuda()
    image_encoder.eval()

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

            print('--- Sampling and R-Precision --- {}_{} --- Epoch: {}'.format(cfg.RUN, version, epoch))

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
            sampling_and_r_precision()
            end_t = time.time()

            print('Time for epoch:', end_t - start_t)

    os.chdir(main_wd)
    print('Loading FID model...')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx])
    if cfg.CUDA:
        fid_model.cuda()

    for version in model_info.VERSIONS.keys():
        version_info = model_info.VERSIONS[version]
        name = '{}_{}'.format(cfg.RUN, version)

        manual_seed = 100
        random.seed(manual_seed)
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(manual_seed)

        # 0, 50, ..., 600
        for epoch in range(0, 601, 50):

            start_t = time.time()
            print('--- IS --- {}_{} --- Epoch: {}'.format(cfg.RUN, version, epoch))
            compute_is()
            print('--- FID --- {}_{} --- Epoch: {}'.format(cfg.RUN, version, epoch))
            compute_fid()
            end_t = time.time()

            print('Time for epoch:', end_t - start_t)

        pprint.pprint(results)
        results_path = os.path.join(output_dir, name, name + '_scores.json')
        with open(results_path, 'w') as outfile:
            json.dump(results[name], outfile)