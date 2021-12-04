from torch_points3d.models.panoptic.pointgroup import PointGroup


def get_model(option, dataset):
    return PointGroup(option, 0, dataset, 0)
    
