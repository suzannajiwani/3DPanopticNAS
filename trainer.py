import torch
import torch.nn as nn
import pytorch_lightning as pl



from config import get_cfg
from torch_points3d1.models.panopticseg.model import get_model

class TrainingModule(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        hparams["conv_type"] = "SPARSE"
        # see config.py for details
        self.hparams.update(hparams)
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg

        # Model
        self.model = get_model(cfg, 0)
        # Losses
        self.losses_fn = nn.ModuleDict()
        self.losses_fn['semantic'] = torch.nn.CrossEntropyLoss()
        # TODO: update instance loss with a custom loss function based on Panoster's description
        self.losses_fn['instance'] = torch.nn.CrossEntropyLoss()
        
        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        lidar_input = batch['lidar']
        labels = batch['labels']

        # Forward pass
        output = self.model(lidar_input)

        #####
        # Loss computation
        #####
        loss = {}
        segmentation_factor = 1 / torch.exp(self.model.segmentation_weight)
        loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
            output['segmentation'], labels['segmentation']
        )
        loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

        centerness_factor = 1 / (2*torch.exp(self.model.centerness_weight))
        loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
            output['instance_center'], labels['centerness']
        )

        offset_factor = 1 / (2*torch.exp(self.model.offset_weight))
        loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
            output['instance_offset'], labels['offset']
        )

        loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight
        loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

        # Metrics
        if not is_train:
            seg_prediction = output['segmentation'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdims=True)
            self.metric_iou_val(seg_prediction, labels['segmentation'])

            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False
            )

            self.metric_panoptic_val(pred_consistent_instance_seg, labels['instance'])

        return output, labels, loss

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar(key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        for key, value in loss.items():
            self.log('val_' + key, value)

        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        # log per class iou metrics
        class_names = ['background', 'dynamic']
        if not is_train:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.training_step_count)
            self.metric_iou_val.reset()

        if not is_train:
            scores = self.metric_panoptic_val.compute()
            for key, value in scores.items():
                for instance_name, score in zip(['background', 'vehicles'], value):
                    if instance_name != 'background':
                        self.logger.experiment.add_scalar(f'val_{key}_{instance_name}', score.item(),
                                                          global_step=self.training_step_count)
            self.metric_panoptic_val.reset()

        self.logger.experiment.add_scalar('segmentation_weight',
                                          1 / (torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('centerness_weight',
                                          1 / (2 * torch.exp(self.model.centerness_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
                                          global_step=self.training_step_count)
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.logger.experiment.add_scalar('flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer
