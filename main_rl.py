import torch
import torch.multiprocessing as mp
from train_rl import train
from dataloader import *
import models
import opts
import my_optim

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))

    torch.manual_seed(opt.seed)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed)

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    model = models.setup(opt)
    model.train()

    if opt.use_cuda:
        model.cuda()

    if opt.async_opt:  # not support ppo for now
        model.share_memory()
        optimizer = my_optim.SharedAdam(model.parameters(),
                                        lr=opt.optim_rl_lr,
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
        rank = 0
        optimizer = None
        train(rank, model, opt, optimizer)
