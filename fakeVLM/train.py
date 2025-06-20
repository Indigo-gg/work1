import os
import time
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    val_opt.jpg_qual = [30]  # Mid-point of typical quality range for validation

    return val_opt



if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    # Initialize TensorBoard writers
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    
    # Training loop
    for epoch in range(opt.niter):
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            # Set input data and optimize parameters
            model.set_input(data)
            model.optimize_parameters()

            # Logging training loss
            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

        # Save model checkpoints
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation phase
        model.eval()
        # Replace: Unpack 4 metrics -> Single accuracy metric
        acc = validate(model.model, val_loader)  # Only accuracy returned
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        print("(Val @ epoch {}) acc: {}".format(epoch, acc))  # Simplified print statement
        
        # Early stopping logic
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()

