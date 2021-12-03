from dataset import PanopDataset

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

class NusceneslidarDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = PanopDataset(
            self._data_path,
            True,
        )
        self.test_dataset = PanopDataset(
            self._data_path,
            False,
        )


    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)