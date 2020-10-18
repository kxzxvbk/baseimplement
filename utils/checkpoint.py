from __future__ import print_function
import os
import shutil
import torch


class Checkpoint:
    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(self, model, optimizer, epoch, max_ans_acc, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.max_ans_acc = max_ans_acc
        self._path = path
        self.flag = 0

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save_according_name(self, experiment_dir, filename, args=None):
        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, filename)
        path = self._path
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'optimizer': self.optimizer,
                    'max_ans_acc': self.max_ans_acc
                    },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))
        return path

    @classmethod
    def load(cls, path):
        print("Loading checkpoints from {}".format(path))
        resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
        model = torch.load(os.path.join(path, cls.MODEL_NAME))
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          max_ans_acc=resume_checkpoint['max_ans_acc'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])

    @classmethod
    def get_certain_checkpoint(cls, experiment_path, filename):
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        return os.path.join(checkpoints_path, filename)
