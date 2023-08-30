from .basevoc import BaseDataset

class voc(BaseDataset):
    def __init__(self, data_dir: str, subset: str, transform):
        super(voc, self).__init__(root=data_dir, subset=subset, transform=transform)
