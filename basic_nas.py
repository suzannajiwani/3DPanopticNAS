import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    else:
        print(OmegaConf.to_yaml(cfg))

    cluster_radii = [0.5, 1, 1.5]
    kernel_sizes = [2,3]
    conv_types = ["SPARSE"]
    for cluster_radius in cluster_radii:
        OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.cluster_radius_search", cluster_radius)
        for kernel_size in kernel_sizes:
            OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.backbone.down_conv.kernel_size", kernel_size)
            OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.backbone.up_conv.kernel_size", kernel_size)
            for conv_type in conv_types:
                OmegaConf.update(cfg, "models.panoptic.PointGroup.conv_type", conv_type)
                OmegaConf.update(cfg, "models.panoptic.PointGroup.scorer_unet.conv_type", conv_type)
                OmegaConf.update(cfg, "models.panoptic.PointGroup.scorer_encoder.conv_type", conv_type)
                
                OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.conv_type", conv_type)
                OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.backbone.conv_type", conv_type)
                OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.scorer_unet.conv_type", conv_type)
                OmegaConf.update(cfg, "models.panoptic.PointGroup-PAPER.scorer_encoder.conv_type", conv_type)
                
                trainer = Trainer(cfg)
                trainer.train()
    
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()