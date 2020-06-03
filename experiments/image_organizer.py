import os
import shutil


if __name__=='__main__':
    directory = "C:\\Users\\45207\Documents\Speciale\Models\DMM-GAN\DMM-STREAM\\netG_epoch_550"

    counter = 0
    subdirs = os.listdir(directory)
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        filenames = os.listdir(subdir_path)
        for filename in filenames:
            src = os.path.join(subdir_path, filename)
            target = os.path.join(directory, '{}_{}'.format(counter, filename))
            shutil.move(src, target)
            counter += 1

    while len(subdirs):
        subdir = subdirs.pop()
        os.rmdir(os.path.join(directory, subdir))

    filenames = os.listdir(directory)
    filenames = list(filter(lambda x: not x.endswith('g2.png'), filenames))
    for filename in filenames:
        os.remove(os.path.join(directory, filename))

    filenames = os.listdir(directory)
    for i in range(10):

        os.makedirs(os.path.join(directory, str(i+1)))
        for _ in range(10):

            filename = filenames.pop()
            src = os.path.join(directory, filename)
            target = os.path.join(directory, str(i+1), filename)
            shutil.move(src, target)

    os.makedirs(os.path.join(directory,'best'))