from torch_points3d.models.panoptic import PointGroup


def get_model(option, dataset):
    return PointGroup({}, 0, dataset, 0)
    
