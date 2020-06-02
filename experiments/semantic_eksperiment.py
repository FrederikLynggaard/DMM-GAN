from importlib import import_module
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json

import random
import torch
import torchvision.transforms as transforms

from datasets import TextDataset, prepare_data

if __name__ == "__main__":

    def get_model_img(model_dir, key):
        cls = key.split('/')[0]
        id = key.split('/')[1]

        filenames = os.listdir(os.path.join(model_dir, cls))
        filenames = list(filter(lambda x: x.startswith(id), filenames))
        file = os.path.join(model_dir, cls, random.choice(filenames))
        return mpimg.imread(file)


    data_dir = "../data/birds"
    base_size = 64

    model_base_dir = ""
    model_other_dir = ""

    # Get data loader
    dataset = TextDataset(data_dir, 'test', base_size=base_size)
    assert dataset
    dataset.norm = transforms.ToTensor()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=True, num_workers=2)

    result = {"base": 0, "other": 0, "undecidable": 0}

    matplotlib.use('TkAgg')
    for step, data in enumerate(dataloader, 0):
        gt_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(2, 1, 1)
        plt.imshow(gt_imgs[-1][0].cpu().permute(1, 2, 0))
        plt.axis("off")

        img_base = get_model_img(model_base_dir, keys[0])
        img_other = get_model_img(model_other_dir, keys[0])

        base_is_left = random.choice([True, False])

        if base_is_left:
            model_imgs = [img_base, img_other]
            model_names = ["base", "other"]
        else:
            model_imgs = [img_other, img_base]
            model_names = ["other", "base"]

        ax = fig.add_subplot(2, 3, 4)
        plt.imshow(model_imgs[0])
        plt.axis("off")

        ax = fig.add_subplot(2, 3, 6)
        plt.imshow(model_imgs[1])
        plt.axis("off")

        fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))

        plt.show(block=False)
        plt.pause(0.0001)
        print("Votes so far: " + str(step))
        while True:
            answer = input("[a = Left][d = Right][0 = Undecidable]: ")
            if answer == "a":
                result[model_names[0]] += 1
                break
            elif answer == "d":
                result[model_names[1]] += 1
                break
            elif answer == "0":
                result["undecidable"] += 1
                break
            else:
                print("Invalid answer!")

        plt.close('all')
        if step == 99:

            json = json.dumps(result)
            f = open("result.json", "w")
            f.write(json)
            f.close()

            break

