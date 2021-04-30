import argparse
import os
import math
import random
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from srwarp import crop
from srwarp import transform
from srwarp import grid
from srwarp import warp
import warp_utils


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


@torch.no_grad()
def eval_warp_psnr(
        loader, model, data_norm=None, eval_type=None, eval_bsize=None,
        verbose=False):

    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    metric_fn = utils.calc_psnr
    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val', ncols=80)
    for batch in pbar:
        gt_ref = batch['inp'].cuda()
        gt = (gt_ref - inp_sub) / inp_div

        m_inv = batch['m']
        m_inv = random.choice(m_inv)
        lr_crop, m = warp_utils.get_input(gt, m_inv, stochastic=False)
        m_inv_comp = transform.inverse_3x3(m)

        sizes = (gt.size(-2), gt.size(-1))
        sizes_source = (lr_crop.size(-2), lr_crop.size(-1))

        grid_raw, yi = grid.get_safe_projective_grid(
            m_inv_comp,
            sizes,
            sizes_source,
        )
        coord = grid.convert_coord(grid_raw, sizes_source)
        coord = coord.view(1, coord.size(0), coord.size(1))
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / gt.size(-1)
        cell[..., 1] *= 2 / gt.size(-2)

        pred = batched_predict(model, lr_crop, coord, cell, 16384)
        pred = pred * inp_div + inp_sub
        pred.clamp_(0, 1)

        ignore_value = -255
        shave = 4
        pred_full = torch.full_like(gt, ignore_value)
        pred_full = pred_full.view(pred_full.size(0), pred_full.size(1), -1)
        pred_full[..., yi] = pred.transpose(-2, -1)
        pred_full = pred_full.view(gt.size())
        mask = (pred_full != ignore_value).float()

        diff = mask * (pred_full - gt_ref)
        diff = diff[..., shave:-shave, shave:-shave]
        gain = mask.nelement() / mask.sum()
        mse = gain.item() * diff.pow(2).mean()
        res = -10 * mse.log10()
        val_res.add(res.item(), 1)

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
