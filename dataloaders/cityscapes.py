from .cars import cars

class cityscapes(cars):
    def __init__(self, data_dir: str, split: str, transform=None, foggy=False):
        super(cityscapes, self).__init__(data_dir, split, transform, foggy)