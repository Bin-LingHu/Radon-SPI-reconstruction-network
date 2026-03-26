import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import torch

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
# model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z

# z_samples_main = model.get_z_random(1, opt.nz)

if __name__ == '__main__':
# test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))


        for nn in range(opt.n_samples + 1):
            encode = nn == 0 and not opt.no_encode

            real_A, fake_B, real_B = model.test(None, encode=encode)

            if nn == 0:
                images = [real_A, real_B, fake_B]
                names = ['input', 'ground truth', 'encoded']
            else:
                images.append(fake_B)
          

        for j in range(opt.batch_size):
            single_image = []
      
            for k in range(len(images)):
                single_image.append(images[k][j:j+1])

            img_path = 'input_%3.3d' % (i * opt.batch_size + j)
            save_images(webpage, single_image, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    webpage.save()

