import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=50, help='total number of training episodes')
        self.parser.add_argument('--show_every', type=int, nargs='?', default=500, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=10, help='How often to print scores')
        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')

        self.parser.add_argument('--batch', type=int, nargs='?', default=256, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.9, help='beta1 for ADAM')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='beta2 for ADAM')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.01, help='leaky Relu Value')
        self.parser.add_argument('--directory', type=str, nargs='?', default='img', help='directory where image gets saved')

        #Discriminator Options
        self.parser.add_argument('--D_hidden', type=int, nargs='+', default=3, help='Depth of Discriminator Network')
        self.parser.add_argument('--D_activation', type=str, nargs='?', default='lrelu', help='Activation function for the discriminator')
        self.parser.add_argument('--D_input_size', type=int, nargs='?', default=784, help='size of input for the discriminator')
        self.parser.add_argument('--D_output_size', type=int, nargs='?', default=1, help='size of output for the discriminator')

        #Generator Options
        self.parser.add_argument('--G_hidden', type=int, nargs='+', default=8, help='Depth of Generator Network')
        self.parser.add_argument('--G_activation', type=str, nargs='?', default='relu', help='Activation function for the generator')
        self.parser.add_argument('--noise_dim', type=int, nargs='?', default=512, help='size of noise input for the generator')
        self.parser.add_argument('--G_output_size', type=int, nargs='?', default=784, help='size of output for the discriminator')
        self.parser.add_argument('--G_out_activation', type=str, nargs='?', default='tanh', help='final output activator')
        self.parser.add_argument('--image_size', type=int, nargs='?', default=28, help='Image Size')
        self.parser.add_argument('--image_channel', type=int, nargs='?', default=1, help='Image channels')
        self.parser.add_argument('--target_is', type=float, nargs='?', default=4.0, help='Target Inception Score')


        #RAY Options
        self.parser.add_argument('--cpu_use', type=int, nargs='?', default=3, help='Number of CPUs to use')
        self.parser.add_argument('--gpu_use', type=float, nargs='?', default=.125, help='Fraction of GPUs to use')
        self.parser.add_argument('--tune_iter', type=int, nargs='?', default=2000, help='number of tuning steps')
        self.parser.add_argument('--num_sample', type=int, nargs='?', default=8, help='number of samples')
        self.parser.add_argument('--perturb_iter', type=int, nargs='?', default=100, help='number of perturb iterations')
        self.parser.add_argument('--train_iterations_per_step', type=int, nargs='?', default=5, help='number of train iter per STEP()')
        self.parser.add_argument('--model_path', type=str, nargs='?', default='/home/youngwook/.ray/models/mnist_cnn.pt', help='directory where inception model gets saved')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt
