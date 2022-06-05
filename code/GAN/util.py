import imageio
import os
import fnmatch
import torch
from torchvision.utils import save_image

def get_filenames(directory, filenames):

    filelist = []

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, filenames):
            filelist.append(file)

    return filelist

def image_to_gif(directory, filenames, duration=0.5, destination=None, gifname=None):
    """
    Given a directory, filename and duration, this function creates a gif using the filename in the directory given
    with a puase of duration seconds between images
    :param directory (str): directory that holds images
    :param filename (list of str)): a list that holds str of names of filenames that will be turned into a gif
    :param duration (float): a pause between images. defaulted to 0.5 second pause
    :param destination (str): destination directory
    :param gifname (str): name for the gif file.
    :return: NA this function simply saves the gif in the directory given
    """

    if destination == None:
        destination = directory

    if gifname == None:
        gifname = 'movie'

    images = []

    for filename in filenames:
        images.append(imageio.imread(os.path.join('%s' %directory, '%s' %filename)))
    imageio.mimsave('%s%s.gif' % (directory, gifname), images, duration=duration)

def sample_noise(batch_size, dim):

    L = -1
    U = 1

    noise = (L - U) * torch.rand((batch_size, dim)) + U

    return noise

def save_images_to_directory(image_tensor, directory, filename):

    directory = directory
    image = torch.from_numpy(image_tensor).data

    save_name = os.path.join('%s' % directory, '%s' % filename)
    save_image(image, save_name)

    return filename
