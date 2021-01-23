import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
from time import time
import argparse
import json
import torch
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from utils.plots import plot_surface, plot_cuts


class ShapeSpaceRunner:

    def run(self):

        print("running")

        for epoch in range(self.startepoch, self.nepochs + 1):

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                self.save_checkpoints(epoch)
                self.plot_validation_shapes(epoch)

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            # start epoch
            before_epoch = time()
            for data_index,(mnfld_pnts, normals, indices) in enumerate(self.train_dataloader):

                mnfld_pnts = mnfld_pnts.cuda()

                if self.with_normals:
                    normals = normals.cuda()

                nonmnfld_pnts = self.sampler.get_points(mnfld_pnts)

                mnfld_pnts = self.add_latent(mnfld_pnts, indices)
                nonmnfld_pnts = self.add_latent(nonmnfld_pnts, indices)

                # forward pass

                mnfld_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()

                mnfld_pred = self.network(mnfld_pnts)
                nonmnfld_pred = self.network(nonmnfld_pnts)

                mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
                nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

                # manifold loss

                mnfld_loss = (mnfld_pred.abs()).mean()

                # eikonal loss

                grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

                loss = mnfld_loss + self.grad_lambda * grad_loss

                # normals loss
                if self.with_normals:
                    normals = normals.view(-1, 3)
                    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                    loss = loss + self.normals_lambda * normals_loss
                else:
                    normals_loss = torch.zeros(1)

                # latent loss

                latent_loss = self.latent_size_reg(indices.cuda())

                loss = loss + self.latent_lambda * latent_loss

                # back propagation

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # print status
                if data_index % self.conf.get_int('train.status_frequency') == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tLatent loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch, data_index * self.batch_size, len(self.ds), 100. * data_index / len(self.train_dataloader),
                               loss.item(), mnfld_loss.item(), grad_loss.item(), latent_loss.item(), normals_loss.item()))

            after_epoch = time()
            print('epoch time {0}'.format(str(after_epoch-before_epoch)))

    def plot_validation_shapes(self, epoch, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            print('plot validation epoch: ', epoch)

            self.network.eval()
            pnts, normals, idx = next(iter(self.eval_dataloader))
            pnts = utils.to_cuda(pnts)

            pnts = self.add_latent(pnts, idx)
            latent = self.lat_vecs[idx[0]]

            shapename = str.join('_', self.ds.get_info(idx))

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         latent=latent,
                         path=self.plots_dir,
                         epoch=epoch,
                         shapename=shapename,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts(points=pnts,
                          decoder=self.network,
                          latent=latent,
                          path=self.plots_dir,
                          epoch=epoch,
                          near_zero=False)

    def __init__(self,**kwargs):

        # config setting

        self.home_dir = os.path.abspath(os.pardir)

        if type(kwargs['conf']) == str:
            self.conf_filename = './shapespace/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        # settings for loading an existing experiment

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = self.timestamp
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.cur_exp_dir))

        self.plots_dir = os.path.join(self.expdir, self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.expdir, self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.expdir, self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.latent_codes_subdir = "LatentCodes"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.latent_codes_subdir))

        self.nepochs = kwargs['nepochs']

        self.batch_size = kwargs['batch_size']

        if self.num_of_gpus > 0:
            self.batch_size *= self.num_of_gpus

        self.parallel = self.num_of_gpus > 1

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.local_sigma = self.conf.get_float('network.sampler.properties.local_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma, self.local_sigma)

        train_split_file = './splits/{0}'.format(kwargs['split_file'])

        with open(train_split_file, "r") as f:
            train_split = json.load(f)

        self.d_in = self.conf.get_int('train.d_in')

        # latent preprocessing

        self.latent_size = self.conf.get_int('train.latent_size')

        self.latent_lambda = self.conf.get_float('network.loss.latent_lambda')
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')

        self.with_normals = self.normals_lambda > 0

        self.ds = utils.get_class(self.conf.get_string('train.dataset'))(split=train_split,
                                                                         with_normals=self.with_normals,
                                                                         dataset_path=self.conf.get_string(
                                                                             'train.dataset_path'),
                                                                         points_batch=kwargs['points_batch'],
                                                                         )

        self.num_scenes = len(self.ds)

        self.train_dataloader = torch.utils.data.DataLoader(self.ds,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=kwargs['threads'], drop_last=True, pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.ds,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=0, drop_last=True)

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=(self.d_in+self.latent_size), **self.conf.get_config('network.inputs'))

        if self.parallel:
            self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        # optimizer and latent settings

        self.startepoch = 0

        self.lat_vecs = torch.zeros(self.num_scenes, self.latent_size).cuda()
        self.lat_vecs.requires_grad_()

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
                {
                    "params": self.lat_vecs,
                    "lr": self.lr_schedules[1].get_learning_rate(0)
                },
            ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            data = torch.load(os.path.join(old_checkpnts_dir, self.latent_codes_subdir, str(kwargs['checkpoint']) + '.pth'))
            self.lat_vecs = data["latent_codes"].cuda()

            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def latent_size_reg(self, indices):
        latents = torch.index_select(self.lat_vecs, 0, indices)
        latent_loss = latents.norm(dim=1).mean()
        return latent_loss

    def get_learning_rate_schedules(self,schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def add_latent(self, points, indices):
        batch_size, num_of_points, dim = points.shape
        points = points.reshape(batch_size * num_of_points, dim)
        latent_inputs = torch.zeros(0).cuda()

        for ind in indices.numpy():
            latent_ind = self.lat_vecs[ind]
            latent_repeat = latent_ind.expand(num_of_points, -1)
            latent_inputs = torch.cat([latent_inputs, latent_repeat], 0)
        points = torch.cat([latent_inputs, points], 1)
        return points

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self,epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "latent_codes": self.lat_vecs},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "latent_codes": self.lat_vecs},
            os.path.join(self.checkpoints_path, self.latent_codes_subdir, "latest.pth"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--points_batch', type=int, default=8000, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='dfaust_setup.conf')
    parser.add_argument('--expname', type=str, default='dfuast_shapespace')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU to use [default: GPU ignore]')
    parser.add_argument('--threads', type=int, default=32, help='num of threads for data loader')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    parser.add_argument('--split', default='dfaust/train_all.json', type=str)

    args = parser.parse_args()

    trainrunner = ShapeSpaceRunner(
            conf=args.conf,
            batch_size=args.batch_size,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=args.gpu,
            threads=args.threads,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            split_file=args.split
    )

    trainrunner.run()
