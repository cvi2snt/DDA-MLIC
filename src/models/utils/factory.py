import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM

def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    else:
        print("model {} not found !!".format(args.model_name))
        exit(-1)

    return model
