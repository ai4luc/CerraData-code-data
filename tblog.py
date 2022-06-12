try:
    from torch.utils.tensorboard import SummaryWriter
    SummaryWriter = SummaryWriter
    is_enabled = True
    logdir = None
    print("=> logging on TensorBoard")
except ImportError:
    is_enabled = False

