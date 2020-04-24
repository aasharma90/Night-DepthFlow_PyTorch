import time, os
from options.train_options import TrainOptions
from dataloader.data_loader import DataLoader
from models.combogan_flow_model import ComboGANflowModel
from tensorboardX import SummaryWriter
import torch

# Get training options
opt         = TrainOptions().parse()
# Create training and validation (if needed) dataloaders
trn_dataset = DataLoader(opt, dataname=opt.dataname, shuffle_data=True,  val_set=False)
print('# Training Image1-Image2 pairs   = %d' % len(trn_dataset))
if not opt.skip_validation:
    raise NotImplementedError

# Create directories for saving training information
total_steps = 0
logs_dir    = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
visuals1_dir= os.path.join(opt.checkpoints_dir, opt.name, 'visuals1')
visuals2_dir= os.path.join(opt.checkpoints_dir, opt.name, 'visuals2')
val_res_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_checks')
logger      = SummaryWriter(logs_dir)

# Initialize the model
model = ComboGANflowModel(opt)
# Update initially if continuing
if opt.which_epoch > 0:
    model.update_hyperparams(opt.which_epoch)

# Validate at epoch0 (if needed)
if not opt.skip_validation:
    raise NotImplementedError

# Being the training and validation process for the given number of epochs!
for epoch in range(1, opt.which_epoch+1):
    epoch_iter = 0
    for i, data in enumerate(trn_dataset):
        total_steps += opt.batchSize
        epoch_iter  += opt.batchSize
# Being the training and validation process for the given number of epochs!
for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(trn_dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter  += opt.batchSize
        model.set_input(data, val_set=False)
        model.optimize_parameters(epoch)

        if total_steps % opt.display_freq == 0:
            model.save_current_visuals(epoch, epoch_iter, visuals1_dir, None)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            model.print_current_errors(epoch, epoch_iter, errors, t)
            model.plot_current_errors(opt.name, ((epoch-1)*len(trn_dataset))+epoch_iter, errors, logger)

    if not opt.skip_validation:
        raise NotImplementedError

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_hyperparams(epoch)
