import time, os
from options.train_options import TrainOptions
from dataloader.data_loader import DataLoader
from models.combogan_stereo_model import ComboGANstereoModel
from tensorboardX import SummaryWriter


opt = TrainOptions().parse()
trn_dataset = DataLoader(opt, dataname=opt.dataname, shuffle_data=True,  val_set=False)
val_dataset = DataLoader(opt, dataname=opt.dataname, shuffle_data=False, val_set=True)
print('# Training Left-Right pairs   = %d' % len(trn_dataset))
print('# Validation Left-Right pairs = %d' % len(val_dataset))

total_steps = 0
logs_dir    = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
visuals1_dir= os.path.join(opt.checkpoints_dir, opt.name, 'visuals1')
visuals2_dir= os.path.join(opt.checkpoints_dir, opt.name, 'visuals2')
val_res_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_checks')
model       = ComboGANstereoModel(opt)
logger      = SummaryWriter(logs_dir)
# Update initially if continuing
if opt.which_epoch > 0:
    model.update_hyperparams(opt.which_epoch)

val_glb_iter  = 0
total_val_loss= 0
for i, data in enumerate(val_dataset):
    val_glb_iter = val_glb_iter+1
    model.set_input(data, val_set=True)
    val_loss = model.perform_validation(0, i, val_res_dir)
    total_val_loss += val_loss
    logger.add_scalar(opt.name+ '/Validation_Disp_iter',  val_loss, val_glb_iter)
logger.add_scalar(opt.name+ '/Validation_Disp_epoch', total_val_loss/(i+1), 0)

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

    total_val_loss = 0
    for i, data in enumerate(val_dataset):
        val_glb_iter = val_glb_iter+1
        model.set_input(data, val_set=True)
        val_loss = model.perform_validation(epoch, i, val_res_dir)
        total_val_loss += val_loss
        logger.add_scalar(opt.name+ '/Validation_Disp_iter',  val_loss, val_glb_iter)
    logger.add_scalar(opt.name+ '/Validation_Disp_epoch', total_val_loss/(i+1), epoch)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_hyperparams(epoch)
