import argparse
from torch import optim

from core.model_conc import CTMNet
from dataset.data_loader import data_loader
from tools.general_utils import *
from core.workflow import *
from core.config import Config

import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eid', type=int, default=-1)
    parser.add_argument('--gpu_id', type=int, nargs='+', default=0)
    parser.add_argument('--yaml_file', type=str, default='configs/demo/mini/5way_5shot.yaml')

    outside_opts = parser.parse_args()

    experiment_name = outside_opts.yaml_file.split("/")[-1].split(".")[0]

    if isinstance(outside_opts.gpu_id, int):
        outside_opts.gpu_id = [outside_opts.gpu_id]  # int -> list

    config = {}
    config['options'] = {
        'ctrl.yaml_file': outside_opts.yaml_file,
        'ctrl.gpu_id': outside_opts.gpu_id
    }
    opts = Config(config['options']['ctrl.yaml_file'], config['options'])
    opts.setup()
    print("-"*100)
    # DATA
    meta_test = None
    train_db_list, val_db_list, _, _ = data_loader(opts)

    # MODEL
    net = CTMNet(opts).to(opts.ctrl.device)
    checkpoints = torch.load(opts.io.model_file, map_location=torch.device('cpu'))

    net.load_state_dict(checkpoints['state_dict'])
    _last_epoch = checkpoints['epoch']
    _last_lr = checkpoints['lr']
    _last_iter = checkpoints['iter']
    opts.io.previous_acc = checkpoints['val_acc']
    opts.logger('\tthis checkpoint is at epoch {}, iter {}, accuracy is {:.4f}\n'.format(
        _last_epoch, _last_iter, opts.io.previous_acc))

    if opts.fsl.evolution:
        which_ind = sum(_last_epoch >= np.array(opts.fsl.epoch_schedule))
    else:
        which_ind = 0
    if _last_iter == opts.ctrl.total_iter_train[which_ind] - 1:
        opts.ctrl.start_epoch = _last_epoch + 1
        opts.ctrl.start_iter = 0
    else:
        opts.ctrl.start_epoch = _last_epoch
        opts.ctrl.start_iter = _last_iter + 1

    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        net.previous_loss_data = checkpoints['loss_data']

    opts.io.saved_epoch = _last_epoch
    opts.io.saved_iter = _last_iter

    which_ind = 0
    curr_shot = opts.fsl.k_shot[0]
    val_db = val_db_list[0]
    eval_length = 50  # opts.ctrl.total_iter_val[0]

    # OPTIM AND LR SCHEDULE
    if opts.train.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
    elif opts.train.optim == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay, momentum=opts.train.momentum)
    elif opts.train.optim == 'rmsprop':
        optimizer = optim.RMSprop(
            net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay, momentum=opts.train.momentum,
            alpha=0.9, centered=True)
    if opts.model.structure == 'original':
        # ignore previous setting
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        opts.train.lr_policy = 'step'
        opts.train.step_size = 100 if not opts.data.use_ori_relation else 3
        opts.train.lr_scheduler = [-1]
        opts.train.lr = 0.001
        opts.train.lr_gamma = 0.5
        opts.train.weight_decay = .0

    accuracy, support_embeddings_lst = test_model2(net, val_db, eval_length, opts, which_ind, curr_shot, optimizer, meta_test)

    pickle.dump(support_embeddings_lst, open("{}_sup_embs.pkl".format(experiment_name), "wb"))

    print("accuracy is", accuracy)


if __name__ == '__main__':
    main()
