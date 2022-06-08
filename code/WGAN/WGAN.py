
# torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

#load mnist dataset and define network
from torchvision import datasets, transforms

import numpy as np
from scipy.stats import entropy

import urllib.request

from filelock import FileLock


# get device.
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

from util import sample_noise, save_images_to_directory, image_to_gif

import os.path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

"""
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, opts, Activation='relu', Norm=True):
        super(Linear, self).__init__()

        steps = [nn.Linear(dim_in, dim_out)]

        if Norm:
            steps.append(nn.BatchNorm1d(dim_out))

        if Activation == 'relu':
            steps.append(nn.ReLU())
        elif Activation == 'lrelu':
            steps.append(nn.LeakyReLU(opts.lrelu_val))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class discriminator(nn.Module):
    def __init__(self, opts):
        super(discriminator, self).__init__()

        steps = [Linear(opts.D_input_size, opts.D_hidden[0], opts, Activation=opts.D_activation, Norm=False)]

        if len(opts.D_hidden) > 1:
            for i in range(len(opts.D_hidden) - 1):
                steps.append(Linear(opts.D_hidden[i], opts.D_hidden[i + 1], opts, Activation=opts.D_activation, Norm=False))

        steps.append(Linear(opts.D_hidden[-1], opts.D_output_size, opts, Activation='', Norm=False))

        self.model = nn.Sequential(*steps)

    def forward(self, x):
        return self.model(x)

class generator(nn.Module):
    def __init__(self, opts):
        super(generator, self).__init__()

        steps = [Linear(opts.noise_dim, opts.G_hidden[0], opts, Activation=opts.G_activation, Norm=False)]

        if len(opts.G_hidden) > 1:
            for i in range(len(opts.G_hidden) - 1):
                steps.append(Linear(opts.G_hidden[i], opts.G_hidden[i + 1], opts, Activation=opts.G_activation))

        steps.append(Linear(opts.G_hidden[-1], opts.G_output_size, opts, Activation=''))

        if opts.G_out_activation == 'tanh':
            final_activation = nn.Tanh()
        elif opts.G_out_activation == 'sigm':
            final_activation = nn.Sigmoid()

        steps.append(final_activation)

        self.model = nn.Sequential(*steps)

    def forward(self, x):
        return self.model(x)
"""


class generator(nn.Module):

    def __init__(self, opts):
        super(generator, self).__init__()

        def block(in_shape, out_shape, norm=True):
            model = [nn.Linear(in_shape, out_shape)]
            if norm:
                model.append(nn.BatchNorm1d(out_shape))
            model.append(nn.LeakyReLU(0.2))
            return model

        self.model = nn.Sequential(
            *block(opts.noise_dim, 128, False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, opts.image_size ** 2),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.model(x)

        return x

class discriminator(nn.Module):

    def __init__(self, opts):
        super(discriminator, self).__init__()

        self.discrim = nn.Sequential(
            nn.Linear(opts.image_size ** 2, 512),
            nn.LeakyReLU(0.20),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.20),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.20),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.discrim(x)

        return x


# inception model
# straight from https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_dcgan_mnist/common.py
class Net(nn.Module):
    """
    LeNet for MNist classification, used for inception_score
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# inception score
def inception_score(imgs, mnist_model_ref, batch_size=32, splits=1):
    N = len(imgs)
    dtype = torch.FloatTensor
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    cm = ray.get(mnist_model_ref)  # Get the mnist model from Ray object store.
    up = nn.Upsample(size=(28, 28), mode="bilinear", align_corners=True).type(dtype)

    def get_pred(x):
        x = up(x)
        x = cm(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    preds = np.zeros((N, 10))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def train(
        G,
        D,
        G_optim,
        D_optim,
        criterion,
        dataloader,
        mnist_model_ref,
        iters,
        opts,
        ):

    x, _ = next(iter(dataloader))

    image_shape = x.shape

    for iter_count, (x, _) in enumerate(dataloader):

        if iter_count == opts.train_iterations_per_step:
            break

        # placeholder
        G_loss = torch.zeros(1)

        G_optim.zero_grad()
        # Get real images and flatten it. This is needed as the model uses nn.Linear()
        x = x.reshape((-1, opts.D_input_size))

        # sample random noise of size opts.noise_dim
        g_fake_seed = sample_noise(len(x), opts.noise_dim)

        # generate images
        fake_images = G(g_fake_seed.to(device))

        if iter_count % opts.n_critic == 0:

            # generate logits for the fake images
            logits_fake = D(fake_images)

            G_loss = -torch.mean(logits_fake)

            G_loss.backward()
            G_optim.step()

        # Train discriminator
        D_optim.zero_grad()

        # discriminate
        logits_real = D(x.to(device))
        logits_fake = D(fake_images.detach())  # detach so the gradient doesn't propagate

        # get the loss
        loss_real = -torch.mean(logits_real)
        loss_fake = torch.mean(logits_fake)

        # combine the losses
        D_loss = loss_real + loss_fake

        D_loss.backward()
        D_optim.step()

        for parameters in D.parameters():
            '''
            clip the weights to be between [-clip_value, clip_value]
            '''
            parameters.data.clamp_(-opts.clip_value, opts.clip_value)

        is_score, is_std = inception_score(fake_images.reshape(image_shape), mnist_model_ref)

        if (iters % opts.print_every == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}, IS:{:.4}'.format(iters, D_loss.item(), G_loss.item(), is_score))

    return D_loss.item(), G_loss.item(), is_score


def main(config, checkpoint_dir=None):

    step = 0
    opts = config['opts']

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.transforms.Normalize((0.5), (0.5))
    ])

    # Init discriminator
    D = discriminator(opts).to(device)

    # Init generator
    G = generator(opts).to(device)

    # Init Optimizer
    D_solver = torch.optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    G_solver = torch.optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    if checkpoint_dir is not None:

        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        D.load_state_dict(checkpoint["D"])
        G.load_state_dict(checkpoint["G"])
        D_solver.load_state_dict(checkpoint["D_solver"])
        G_solver.load_state_dict(checkpoint["G_solver"])
        step = checkpoint["step"]

        if "D_lr" in config:
            for param_group in D_solver.param_groups:
                param_group["lr"] = config["D_lr"]
        if "G_lr" in config:
            for param_group in G_solver.param_groups:
                param_group["lr"] = config["G_lr"]
        if "batch" in config:
            opts.batch = config['batch']
        if "n_critic" in config:
            opts.n_critic = config['n_critic']

    with FileLock(os.path.expanduser("~/.data.lock")):
        # Download and load the training data
        trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch, shuffle=True, drop_last=True)


    if not os.path.isdir(opts.directory):
        os.mkdir(opts.directory)

    criterion = nn.MSELoss()

    is_score = 0
    while is_score < opts.target_is:

        lossD, lossG, is_score = train(G,
                                       D,
                                       G_solver,
                                       D_solver,
                                       criterion,
                                       trainloader,
                                       config["mnist_model_ref"],
                                       step,
                                       opts,
                                       )

        step += 1
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                {
                    "D": D.state_dict(),
                    "G": G.state_dict(),
                    "D_solver": D_solver.state_dict(),
                    "G_solver": G_solver.state_dict(),
                    "step": step,
                },
                path,
            )
        tune.report(iters=step, lossg=lossG, lossd=lossD, is_score=is_score)

def show_result(config, checkpoint_dir):

    opts = config['opts']

    filelist = []
    
    # Init generator
    G = generator(opts).to(device)

    if checkpoint_dir is not None:

        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        G.load_state_dict(checkpoint["G"])

    for i in range(5):
        # sample random noise of size opts.noise_dim
        g_fake_seed = sample_noise(opts.batch, opts.noise_dim)
        
        # generate images
        fake_images = G(g_fake_seed.to(device)).reshape(-1, opts.image_channel, opts.image_size, opts.image_size)
        '''filename used for saving the image'''
        filelist.append(save_images_to_directory(fake_images.data.cpu().numpy(), opts.directory, 'generated_image_%s.png' % i))
        
    # create a gif
    image_to_gif(opts.directory + '/', filelist, duration=1)



def pbt(opts):

    ray.init()

    # Download a pre-trained MNIST model for inception score calculation.
    # This is a tiny model (<100kb).
    if not os.path.exists(opts.model_path):
        print("downloading model")
        os.makedirs(os.path.dirname(opts.model_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ray-project/ray/raw/master/python/ray/tune/"
            "model/mnist_cnn.pt",
            opts.model_path,
        )

    # load the pretrained mnist classification model for inception_score
    mnist_cnn = Net()
    mnist_cnn.load_state_dict(torch.load(opts.model_path))
    mnist_cnn.eval()
    # Put the model in Ray object store.
    mnist_model_ref = ray.put(mnist_cnn)

    # PBT scheduler
    scheduler = PopulationBasedTraining(
        perturbation_interval=opts.perturb_iter,
        hyperparam_mutations={
            # distribution for resampling
            "G_lr": lambda: np.random.uniform(1e-3, 1e-5),
            "D_lr": lambda: np.random.uniform(1e-3, 1e-5),
        },
    )

    config = {
            "opts": opts,
            "use_gpu": True,
            "G_lr": tune.choice([0.00005, 0.00001, 0.000025]),
            "D_lr": tune.choice([0.00005, 0.00001, 0.000025]),
            "mnist_model_ref": mnist_model_ref,
        }

    reporter = CLIReporter(
        metric_columns=["iters", "lossg", "lossd", "is_score"])

    analysis = tune.run(
        main,
        name="test",
        scheduler=scheduler,
        resources_per_trial={"cpu": opts.cpu_use, "gpu": opts.gpu_use},
        verbose=1,
        stop={
            "training_iteration": opts.tune_iter,
        },
        metric="is_score",
        mode="max",
        num_samples=opts.num_sample,
        progress_reporter=reporter,
        config=config
    )

    all_trials = analysis.trials
    checkpoint_paths = [
        os.path.join(analysis.get_best_checkpoint(t), "checkpoint")
        for t in all_trials
    ]

    best_trial = analysis.get_best_trial("is_score", "max", "last-5-avg")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="is_score")

    show_result(config, best_checkpoint)

if __name__ == '__main__':
    # options
    from options import options

    options = options()
    opts = options.parse()

    pbt(opts)

