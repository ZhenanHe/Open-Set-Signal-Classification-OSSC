import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def validate_var(Dataset, model, criterion, epoch, writer, device, args):
    """
    Evaluates/validates the model
    """

    # initialize average meters to accumulate values
    cl_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # evaluate the entire validation dataset
    with torch.no_grad():
        for i, (inp, target) in enumerate(Dataset.val_loader):
            inp = inp.to(device)
            target = target.to(device)

            # compute output
            output_samples, mu, std = model(inp)

            # compute loss
            cl_loss, kld_loss = criterion(output_samples, target, mu, std, device)

            # take mean to compute accuracy
            # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
            output = torch.mean(output_samples, dim=0)

            # measure and update accuracy
            prec1 = accuracy(output, target)[0]
            top1.update(prec1.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update the respective loss values. To be consistent with values reported in the literature we scale
            # our normalized losses back to un-normalized values.
            # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
            # across potential weighting terms.
            cl_losses.update(cl_loss.item() * model.module.num_classes, inp.size(0))
            kld_losses.update(kld_loss.item() * model.module.latent_dim, inp.size(0))
            losses.update((cl_loss + kld_loss).item(), inp.size(0))

            # Print progress
            if i % args.print_freq == 0:
                print('Validate: [{0}][{1}/{2}]\t' 
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                       epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=cl_losses,
                       top1=top1, KLD_loss=kld_losses))

    # TensorBoard summary logging
    writer.add_scalar('validation/val_precision@1', top1.avg, epoch)
    writer.add_scalar('validation/val_class_loss', cl_losses.avg, epoch)
    writer.add_scalar('validation/val_average_loss', losses.avg, epoch)
    writer.add_scalar('validation/val_KLD', kld_losses.avg, epoch)

    print(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    return top1.avg, losses.avg
