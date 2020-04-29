
# load dataloader('test')
# load enkel model + vægte
# Kør forward pass
# gem resultat

from __future__ import print_function

import errno

from PIL import Image
from torch.autograd import Variable

from config import cfg, cfg_from_file

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
import torchvision.transforms as transforms


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


def evaluate(netG, image_encoder, text_encoder, dataloader, output_dir, name):
    for step, data in enumerate(dataloader, 0):

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
            fullpath = '%s.png' % s_tmp
            os.chdir(main_wd)
            im.save(fullpath)
            os.chdir(model_wd)


# def evaluate():
#     if cfg.TRAIN.NET_G == '':
#         print('Error: the path for morels is not found!')
#     else:
#         if split_dir == 'test':
#             split_dir = 'valid'
#         # Build and load the generator
#         if cfg.GAN.B_DCGAN:
#             netG = G_DCGAN()
#         else:
#             netG = G_NET()
#         netG.apply(weights_init)
#         netG.cuda()
#         netG.eval()
#
#         # load text encoder
#         text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
#         state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
#         text_encoder.load_state_dict(state_dict)
#         print('Load text encoder from:', cfg.TRAIN.NET_E)
#         text_encoder = text_encoder.cuda()
#         text_encoder.eval()
#
#         # load image encoder
#         image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
#         img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
#         state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
#         image_encoder.load_state_dict(state_dict)
#         print('Load image encoder from:', img_encoder_path)
#         image_encoder = image_encoder.cuda()
#         image_encoder.eval()
#
#         batch_size = self.batch_size
#         nz = cfg.GAN.Z_DIM
#         noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
#         noise = noise.cuda()
#
#         model_dir = cfg.TRAIN.NET_G
#         state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
#         # state_dict = torch.load(cfg.TRAIN.NET_G)
#         netG.load_state_dict(state_dict)
#         print('Load G from: ', model_dir)
#
#         # the path to save generated images
#         s_tmp = model_dir[:model_dir.rfind('.pth')]
#         save_dir = '%s/%s' % (s_tmp, split_dir)
#         mkdir_p(save_dir)
#
#         cnt = 0
#         R_count = 0
#         R = np.zeros(30000)
#         cont = True
#         for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
#             if (cont == False):
#                 break
#             for step, data in enumerate(self.data_loader, 0):
#                 cnt += batch_size
#                 if (cont == False):
#                     break
#                 if step % 100 == 0:
#                     print('cnt: ', cnt)
#                 # if step > 50:
#                 #     break
#
#                 imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
#
#                 hidden = text_encoder.init_hidden(batch_size)
#                 # words_embs: batch_size x nef x seq_len
#                 # sent_emb: batch_size x nef
#                 words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
#                 words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
#                 mask = (captions == 0)
#                 num_words = words_embs.size(2)
#                 if mask.size(1) > num_words:
#                     mask = mask[:, :num_words]
#
#                 #######################################################
#                 # (2) Generate fake images
#                 ######################################################
#                 noise.data.normal_(0, 1)
#                 fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
#                 for j in range(batch_size):
#                     s_tmp = '%s/single/%s' % (save_dir, keys[j])
#                     folder = s_tmp[:s_tmp.rfind('/')]
#                     if not os.path.isdir(folder):
#                         # print('Make a new folder: ', folder)
#                         mkdir_p(folder)
#                     k = -1
#                     # for k in range(len(fake_imgs)):
#                     im = fake_imgs[k][j].data.cpu().numpy()
#                     # [-1, 1] --> [0, 255]
#                     im = (im + 1.0) * 127.5
#                     im = im.astype(np.uint8)
#                     im = np.transpose(im, (1, 2, 0))
#                     im = Image.fromarray(im)
#                     fullpath = '%s_s%d_%d.png' % (s_tmp, k, ii)
#                     im.save(fullpath)
#
#                 _, cnn_code = image_encoder(fake_imgs[-1])
#
#                 for i in range(batch_size):
#                     mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])
#                     hidden = text_encoder.init_hidden(99)
#                     _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
#                     rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
#                     ### cnn_code = 1 * nef
#                     ### rnn_code = 100 * nef
#                     scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
#                     cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
#                     rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
#                     norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
#                     scores0 = scores / norm.clamp(min=1e-8)
#                     if torch.argmax(scores0) == 0:
#                         R[R_count] = 1
#                     R_count += 1
#
#                 if R_count >= 30000:
#                     sum = np.zeros(10)
#                     np.random.shuffle(R)
#                     for i in range(10):
#                         sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
#                     R_mean = np.average(sum)
#                     R_std = np.std(sum)
#                     print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
#                     cont = False

if __name__ == "__main__":
    main_wd = os.getcwd()
    cfg_from_file('experiments_bird.yml')

    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    mkdir_p(output_dir)  # TODO remove from loop


    for x in cfg.MODELS.keys():
        os.chdir(main_wd)
        model_info = cfg.MODELS[x]
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
            # image_encoder = CNN_ENCODER(cfg_x.TEXT.EMBEDDING_DIM)
            # state_dict = torch.load(model_info.IMAGE_ENCODER_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
            # image_encoder.load_state_dict(state_dict)
            # print('Load image encoder from:', model_info.IMAGE_ENCODER_WEIGHTS_PATH)
            # image_encoder = image_encoder.cuda()
            # image_encoder.eval()

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

            image_encoder = None

            start_t = time.time()
            evaluate(netG, image_encoder, text_encoder, dataloader, output_dir, '{}_{}'.format(x, version))
            end_t = time.time()
            print('Total time for training:', end_t - start_t)
