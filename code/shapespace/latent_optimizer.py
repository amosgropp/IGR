import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import torch
from model.network import gradient
from model.sample import Sampler


def adjust_learning_rate(initial_lr, optimizer, iter):
    adjust_lr_every = 400
    lr = initial_lr * ((0.1) ** (iter // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def optimize_latent(points, normals, conf, num_of_iterations, network, lr=1.0e-2):

    latent_size = conf.get_int('train.latent_size')
    global_sigma = conf.get_float('network.sampler.properties.global_sigma')
    local_sigma = conf.get_float('network.sampler.properties.local_sigma')
    sampler = Sampler.get_sampler(conf.get_string('network.sampler.sampler_type'))(global_sigma, local_sigma)

    latent_lambda = conf.get_float('network.loss.latent_lambda')

    normals_lambda = conf.get_float('network.loss.normals_lambda')

    grad_lambda = conf.get_float('network.loss.lambda')

    num_of_points, dim = points.shape

    latent = torch.ones(latent_size).normal_(0, 1 / latent_size).cuda()
    # latent = torch.zeros(latent_size).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    for i in range(num_of_iterations):

        sample = sampler.get_points(points.unsqueeze(0)).squeeze()

        latent_all = latent.expand(num_of_points, -1)
        surface_pnts = torch.cat([latent_all, points], dim=1)

        sample_latent_all = latent.expand(sample.shape[0], -1)
        nonsurface_pnts = torch.cat([sample_latent_all, sample], dim=1)

        surface_pnts.requires_grad_()
        nonsurface_pnts.requires_grad_()

        surface_pred = network(surface_pnts)
        nonsurface_pred = network(nonsurface_pnts)

        surface_grad = gradient(surface_pnts, surface_pred)
        nonsurface_grad = gradient(nonsurface_pnts, nonsurface_pred)

        surface_loss = torch.abs(surface_pred).mean()
        grad_loss = torch.mean((nonsurface_grad.norm(2, dim=-1) - 1).pow(2))
        normals_loss = ((surface_grad - normals).abs()).norm(2, dim=1).mean()
        latent_loss = latent.abs().mean()
        loss = surface_loss + latent_lambda * latent_loss + normals_lambda * normals_loss + grad_lambda * grad_loss

        adjust_learning_rate(lr, optimizer, i)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        print('latent loss iter {0}:{1}'.format(i, loss.item()))

    return latent.unsqueeze(0)
