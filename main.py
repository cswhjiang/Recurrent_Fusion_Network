import torch
import torch.multiprocessing as mp
from train import train
from dataloader import *
import models
import opts
import my_optim

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_num_params(model):
    num = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n * s
        num += n
    return num


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))

    torch.manual_seed(opt.seed)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed)

    loader = DataLoader(opt)  # not used in training procedure, just used to set vocab_size and seq_length
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    model = models.setup(opt)
    model.train()
    num_parameter = get_num_params(model)
    print('number of parameters: ' + str(num_parameter))

    if opt.async_opt:
        if opt.use_cuda:
            model.cuda()
        model.share_memory()
        optimizer = my_optim.SharedAdam(model.parameters(),
                                        lr=opt.optim_lr,
                                        betas=(opt.optim_adam_beta1, opt.optim_adam_beta2),
                                        weight_decay=opt.optim_weight_decay)
        optimizer.share_memory()
        processes = []
        for rank in range(opt.num_processes):
            p = mp.Process(target=train, args=(rank, model, opt, optimizer))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        if opt.use_cuda:
            model.cuda()
        rank = 0
        optimizer = None
        train(rank, model, opt, optimizer)
