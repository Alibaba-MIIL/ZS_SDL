import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL
# from ..resnext.modules import GlobalAvgPool2dResNext
from ..utils.global_avg_pooling import GlobalAvgPool2dResNext
from torch.nn import Linear


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    # Make sure no bottlneck_head is used
    model_params['args'].do_bottleneck_head = False
    model_params['args'].bottleneck_features = None

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    # Add global_avg_pool and embedding matrix
    num_features = model.num_features
    model = model.body
    model.add_module('global_avg_pool', GlobalAvgPool2dResNext())
    model.add_module('embedding', Linear(num_features, args.num_rows * args.wordvec_dim, bias=False))

    return model
