import torch
import shutil
import os


def save_checkpoint(state, is_best, file_path, file_name='checkpoint.pth.tar'):
    """
    Saves the current state of the model. Does a copy of the file
    in case the model performed better than previously.
    """

    save_path = os.path.join(file_path, file_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(file_path, 'model_best.pth.tar'))
