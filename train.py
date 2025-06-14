import os
import sys
import torch
import logging
import numpy as np

from utils.dataloader import get_dataloader
from evaluations import *
from generic_train import train_model_generic

from models.kans import MLP_KAN_FACTORY
from layers.kan_conv import CONV_KAN_FACTORY

from models.kan_vgg import *
from models.kan_alexnet import *
from models.kan_efficientnet import *
from models.kan_efficientnetv2 import *
from models.kan_mobilenet import *
from models.kan_mobilenetv2 import *
from models.kan_mobilenetv3 import *

import argparse

NORM_LAYER = {
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "GroupNorm": nn.GroupNorm,
    "RMSNorm": nn.RMSNorm,
    "None": None
}

BASE_ACTIVATION = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "hardswish": nn.Hardswish,
    "None": None
}

parser = argparse.ArgumentParser(description="Training.")
parser.add_argument("--seed", default=42, type=int, help="Seed for experiments")
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100'], help='Choose the dataset to use for the experiment')
parser.add_argument('--batch_size', type=int, default=64, help='Choose the batch size for experiments.')
parser.add_argument('--data_path', type=str, default="./data", help='Choose dataset path.')
parser.add_argument('--epoch', type=int, default=20, help="Epoch used to evaluate")
parser.add_argument('--model', type=str, default="FlexibleAlexNet",
                    choices=['VGGKAN', 'FlexibleAlexNet', 'EfficientNetKAN', 'EfficientNetV2KAN',
                             'AlexNetKAN', 'MobileNetV2KAN', 'MobileNetV1KAN', 'MobileNetV3KAN'],
                    help="Choose model to work with.")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay")
parser.add_argument('--gamma', type=float, default=0.8, help="Gamma for scheduler")
parser.add_argument('--imagenet_preprocessing', action='store_true', help='Use ImageNet preprocessing (resize/normalize) in dataloader.')
parser.add_argument('--results_dir', type=str, default="./results", help='Directory to save models and potentially logs/plots')
parser.add_argument('--num_workers', type=int, default=1, help='Choose number of workers')
parser.add_argument('--use_cuda', action='store_true', help="Whether to use cuda or not.")

parser.add_argument('--arch', type=str, help='Choose variation architecture of model.')
parser.add_argument('--classifier_type', type=str, default="Linear", help='Classifier classifier type.')
parser.add_argument('--conv_type', type=str, default="kanconv", choices=['kanconv', 'conv'], help='Convolutional layer type: kan or standard conv (relevant if model is VGGKAN).')
parser.add_argument('--kan_conv', type=str,  default="KAN", choices=list(CONV_KAN_FACTORY.keys()), help='Specific KAN convolution type if conv_type is kan (relevant if model is VGGKAN).')
parser.add_argument('--kan_classifier', type=str, default="KAN", choices=list(MLP_KAN_FACTORY.keys()), help='Specific KAN classifier type if classifier_type is kan related.')
parser.add_argument('--norm_layer', type=str, default="BatchNorm2d", choices=list(NORM_LAYER.keys()), help='Normalization layer type.')
parser.add_argument('--kan_norm_layer', type=str, default="BatchNorm2d",  choices=list(NORM_LAYER.keys()), help='Normalization layer type.')
parser.add_argument('--norm_affine', action='store_true', help='Use affine parameters in normalization layer.')
parser.add_argument('--dropout_conv', type=float, default=0.0, help='Dropout rate for KAN convolutional layers.')
parser.add_argument('--dropout_linear', type=float, default=0.5, help='Dropout rate for classifier head layers.')
parser.add_argument('--grid_size', type=int, default=5, help='Grid size for spline KANs.')
parser.add_argument('--spline_order', type=int, default=3, help='Spline order for spline KANs.')
parser.add_argument('--l1_decay', type=float, default=0.0, help='L1 decay for KAN layer coefficients.')
parser.add_argument('--groups', type=int, default=1, help='Number of groups for convolutions.')
parser.add_argument('--degree', type=int, default=3, help='Degree for polynomial KANs.') 

parser.add_argument('--classifier_dropout', type=float, default=None, help='Override dropout_linear specifically for KAN MLP classifier layer.')
parser.add_argument('--classifier_grid_size', type=int, default=None, help='Override grid_size specifically for KAN MLP classifier layer.')
parser.add_argument('--classifier_spline_order', type=int, default=None, help='Override spline_order specifically for KAN MLP classifier layer.')
parser.add_argument('--classifier_l1_decay', type=float, default=None, help='Override l1_decay specifically for KAN MLP classifier layer.')
parser.add_argument('--classifier_degree', type=int, default=None, help='Override degree specifically for KAN MLP classifier layer.')
parser.add_argument('--classifier_base_activation', type=str, default='None', help='Override base_activation specifically for KAN MLP classifier layer.')

# parser.add_argument('--classifier_grid_range', type=float, default=0.0, help='Override l1_decay specifically for KAN MLP head layers.')

parser.add_argument('--width_scale', type=float, default=1, help='Width multiplier for VGG channels.')

parser.add_argument('--stochastic_depth_prob', type=float, default=0.2, help='Stochastic depth probability (EfficientNet).')
parser.add_argument('--replace_depthwise', action='store_true', help='Replace depthwise convs with KAN convs in EfficientNet.')

args = parser.parse_args()

root_path = './'
data_path = os.path.join(root_path, args.data_path)
dataset_path = os.path.join(data_path, args.dataset)
results_path = os.path.join(root_path, args.results_dir)
model_save_path = os.path.join(results_path, args.dataset, "models")
log_save_path = os.path.join(results_path, args.dataset, "logs")

os.makedirs(data_path, exist_ok = True)
os.makedirs(dataset_path, exist_ok = True)
os.makedirs(results_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)

log_file_name = f"{args.model}_{args.classifier_type}_{args.dataset}_seed{args.seed}_train.log"
log_file_path = os.path.join(log_save_path, log_file_name)

file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO) 
file_formatter = logging.Formatter(
    '%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').addHandler(stream_handler)

logger.info("--- Start Training Script ---")
logger.info(f"Logging to console and to file: {log_file_path}")
logger.info("Arguments received:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")
logger.info(f"Setting random seed to {args.seed}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available() and args.use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Loading dataset: {args.dataset}")
if args.imagenet_preprocessing:
    input_shape = [3, 224, 224]
    expected_output_shape = (7, 7)
    train_loader, test_loader, train_set, test_set, classes = get_dataloader(args.dataset, args.batch_size, args.data_path, args.num_workers, imagenet_preprocessing=args.imagenet_preprocessing)
else:
    if args.dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        input_shape = [3, 32, 32]
    else:
        input_shape = [1, 28, 28]
    expected_output_shape = (1, 1)
    train_loader, test_loader, train_set, test_set, classes = get_dataloader(args.dataset, args.batch_size, args.data_path, args.num_workers)
num_classes = len(classes)

model_name = args.model
model = None
if args.model == "VGGKAN":
    logger.info(f"Configuring VGGKAN model (Type: {args.arch})...")
    model = vggkan(
        input_channels=input_shape[0],
        num_classes=num_classes,
        conv_type=args.conv_type,
        kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
        kan_classifier=args.kan_classifier if args.classifier_type in ['VGGKAN', 'KAN', 'HiddenKAN'] else None,
        dropout=args.dropout_conv,
        l1_decay=args.l1_decay,
        dropout_linear=args.dropout_linear, 
        arch=args.arch,
        classifier_type=args.classifier_type,
        expected_feature_shape=expected_output_shape,
        width_scale=args.width_scale,
        affine=args.norm_affine,
        norm_layer=NORM_LAYER.get(args.norm_layer),
        kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    ).to(device)

elif args.model == "EfficientNetKAN":
    logger.info(f"Configuring EfficientNetKAN model (Arch: {args.arch})...")
    if not args.imagenet_preprocessing:
        logger.warning("EfficientNet models are typically trained on ImageNet sizes (224x224+). Performance might be affected by using smaller dataset sizes without ImageNet preprocessing.")
    if 'small' in args.arch:
        model = efficientnet_kan_small(
            arch=args.arch,
            num_classes=num_classes,
            in_channels=input_shape[0],
            conv_type=args.conv_type,
            kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
            replace_depthwise=args.replace_depthwise,
            classifier_type=args.classifier_type,
            kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY or args.classifier_type == 'HiddenKAN' else None,
            norm_layer=NORM_LAYER.get(args.norm_layer),
            kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
            affine=args.norm_affine,
            dropout=args.dropout_linear, 
            stochastic_depth_prob=args.stochastic_depth_prob,
            l1_decay=args.l1_decay,
            groups=args.groups if args.model == 'VGGKAN' else 1,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            conv_dropout=args.dropout_conv,
            classifier_spline_order=args.classifier_spline_order, 
            classifier_grid_size=args.classifier_grid_size,       
            classifier_dropout=args.classifier_dropout,         
            classifier_l1_decay=args.classifier_l1_decay,
            classifier_degree=args.classifier_degree, 
            classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
        ).to(device)
    else:
        model = efficientnet_kan(
            arch=args.arch,
            num_classes=num_classes,
            in_channels=input_shape[0],
            conv_type=args.conv_type,
            kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
            replace_depthwise=args.replace_depthwise,
            classifier_type=args.classifier_type,
            kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY or args.classifier_type == 'HiddenKAN' else None,
            norm_layer=NORM_LAYER.get(args.norm_layer),
            kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
            affine=args.norm_affine,
            dropout=args.dropout_linear, 
            stochastic_depth_prob=args.stochastic_depth_prob,
            l1_decay=args.l1_decay,
            groups=args.groups if args.model == 'VGGKAN' else 1,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            conv_dropout=args.dropout_conv, 
            classifier_spline_order=args.classifier_spline_order, 
            classifier_grid_size=args.classifier_grid_size,       
            classifier_dropout=args.classifier_dropout,         
            classifier_l1_decay=args.classifier_l1_decay,
            classifier_degree=args.classifier_degree,
            classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
        ).to(device)

elif args.model == "EfficientNetV2KAN":
    logger.info(f"Configuring EfficientNetV2KAN model (Arch: {args.arch})...")
    if not args.imagenet_preprocessing:
        logger.warning("EfficientNetV2 models are typically trained on ImageNet sizes (e.g., 224x224+). Performance might be affected by using smaller dataset sizes without ImageNet preprocessing.")
    if args.arch not in ['s', 'm', 'l', 'tiny', 'kan_tiny']:
         logger.warning(f"EfficientNetV2 arch '{args.arch}' not explicitly supported in config (s, m, l). Attempting anyway...")
         raise ValueError(f"Unsupported EfficientNetV2 arch: {args.arch}. Choose from ['s', 'm', 'l', 'tiny', 'kan_tiny'].")

    if args.arch not in ['tiny', 'kan_tiny']:
        model = efficientnetv2_kan(
            arch=args.arch,
            num_classes=num_classes,
            in_channels=input_shape[0],

            conv_type=args.conv_type,
            kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
            replace_depthwise=args.replace_depthwise, 
            classifier_type=args.classifier_type,
            kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY or args.classifier_type == 'HiddenKAN' else None,
            norm_layer=NORM_LAYER.get(args.norm_layer),
            kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
            affine=args.norm_affine,
            dropout=args.dropout_linear, 
            stochastic_depth_prob=args.stochastic_depth_prob, 

            l1_decay=args.l1_decay,
            groups=1,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            conv_dropout=args.dropout_conv, 
            
            classifier_spline_order=args.classifier_spline_order, 
            classifier_grid_size=args.classifier_grid_size,       
            classifier_dropout=args.classifier_dropout,         
            classifier_l1_decay=args.classifier_l1_decay,
            classifier_degree=args.classifier_degree,
            classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
        ).to(device)
    else:
        model = efficientnetv2_kan_small(
            arch=args.arch,
            num_classes=num_classes,
            in_channels=input_shape[0],
            conv_type=args.conv_type,
            kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
            replace_depthwise=args.replace_depthwise, 
            classifier_type=args.classifier_type,
            kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY or args.classifier_type == 'HiddenKAN' else None,
            norm_layer=NORM_LAYER.get(args.norm_layer),
            kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
            affine=args.norm_affine,
            dropout=args.dropout_linear, 
            stochastic_depth_prob=args.stochastic_depth_prob, 

            l1_decay=args.l1_decay,
            groups=1,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            conv_dropout=args.dropout_conv, 

            classifier_spline_order=args.classifier_spline_order, 
            classifier_grid_size=args.classifier_grid_size,       
            classifier_dropout=args.classifier_dropout,         
            classifier_l1_decay=args.classifier_l1_decay,
            classifier_degree=args.classifier_degree,
            classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
        ).to(device)
elif args.model == "AlexNetKAN":
    logger.info(f"Configuring AlexNetKAN model...")
    if not args.imagenet_preprocessing and input_shape[1] < 67: # AlexNet conv1 needs ~67px for pool3 output > 0
         logger.warning("AlexNet architecture performs best with larger input sizes (e.g., 224x224 via --imagenet_preprocessing). Performance might be suboptimal on small images like CIFAR/MNIST without it.")
    model = alexnet_kan(
        num_classes=num_classes,
        input_channels=input_shape[0],
        dropout=args.dropout_linear,
        arch=args.arch,

        conv_type=args.conv_type,
        kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
        classifier_type=args.classifier_type,
        kan_classifier=args.kan_classifier if args.classifier_type in CONV_KAN_FACTORY or 'KAN' in args.classifier_type else None,

        norm_layer=NORM_LAYER.get(args.norm_layer),
        kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
        affine=args.norm_affine,
        
        l1_decay=args.l1_decay,
        groups=args.groups,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        degree=args.degree,
        conv_dropout=args.dropout_conv, 
        
        classifier_spline_order=args.classifier_spline_order, 
        classifier_grid_size=args.classifier_grid_size,       
        classifier_dropout=args.classifier_dropout,         
        classifier_l1_decay=args.classifier_l1_decay,
        classifier_degree=args.classifier_degree,
        classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
        # classifier_grid_range=,
    ).to(device)
elif args.model == "MobileNetV1KAN":
    logger.info(f"Configuring MobileNetV1KAN model (Width Multiplier: {args.width_scale})...")
    model = mobilenet_v1_kan(
        num_classes=num_classes,
        input_channels=input_shape[0],
        width_mult=args.width_scale, 
        dropout=args.dropout_linear,
        conv_type=args.conv_type,
        kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
        replace_depthwise=args.replace_depthwise,
        classifier_type=args.classifier_type,
        kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY else None, 
        norm_layer=NORM_LAYER.get(args.norm_layer),
        kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
        affine=args.norm_affine,
        base_activation=BASE_ACTIVATION[args.classifier_base_activation], 
        l1_decay=args.l1_decay,
        groups=args.groups, 
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        degree=args.degree,
        conv_dropout=args.dropout_conv,
        classifier_spline_order=args.classifier_spline_order,
        classifier_grid_size=args.classifier_grid_size,
        classifier_l1_decay=args.classifier_l1_decay,
        classifier_dropout=args.classifier_dropout,
        classifier_degree=args.classifier_degree,
        classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation]
    ).to(device)
elif args.model == "MobileNetV2KAN": 
    logger.info(f"Configuring MobileNetV2KAN model (Width Multiplier: {args.width_scale})...")
    model = mobilenet_v2_kan(
        num_classes=num_classes,
        arch=args.arch,
        input_channels=input_shape[0],
        width_mult=args.width_scale, 
        dropout=args.dropout_linear,
        conv_type=args.conv_type,
        kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
        replace_depthwise=args.replace_depthwise, 
        classifier_type=args.classifier_type,
        kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY else None,
        norm_layer=NORM_LAYER.get(args.norm_layer),
        kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
        affine=args.norm_affine,
        l1_decay=args.l1_decay,
        groups=args.groups,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        degree=args.degree,
        conv_dropout=args.dropout_conv,
        classifier_spline_order=args.classifier_spline_order,
        classifier_grid_size=args.classifier_grid_size,
        classifier_l1_decay=args.classifier_l1_decay,
        classifier_dropout=args.classifier_dropout,
        classifier_degree=args.classifier_degree,
    ).to(device)

elif args.model == "MobileNetV3KAN":
    logger.info(f"Configuring MobileNetV3KAN model (Arch: {args.arch}, Width Multiplier: {args.width_scale})...")
    if not args.arch or args.arch not in ['large', 'small']:
        logger.error(f"MobileNetV3 requires --arch to be 'large' or 'small'. Got: {args.arch}")
        sys.exit(1)
    model = mobilenet_v3_kan(
        arch=args.arch,
        num_classes=num_classes,
        input_channels=input_shape[0],
        width_mult=args.width_scale, 
        dropout=args.dropout_linear,
        conv_type=args.conv_type,
        kan_conv=args.kan_conv if args.conv_type == 'kanconv' else None,
        replace_depthwise=args.replace_depthwise,
        classifier_type=args.classifier_type,
        kan_classifier=args.kan_classifier if args.classifier_type in MLP_KAN_FACTORY else None,
        norm_layer=NORM_LAYER.get(args.norm_layer),
        kan_norm_layer=NORM_LAYER.get(args.kan_norm_layer),
        affine=args.norm_affine,
        base_activation=BASE_ACTIVATION[args.classifier_base_activation],
        l1_decay=args.l1_decay,
        groups=args.groups, 
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        degree=args.degree,
        conv_dropout=args.dropout_conv,
        classifier_spline_order=args.classifier_spline_order,
        classifier_grid_size=args.classifier_grid_size,
        classifier_l1_decay=args.classifier_l1_decay,
        classifier_dropout=args.classifier_dropout,
        classifier_degree=args.classifier_degree,
        classifier_base_activation=BASE_ACTIVATION[args.classifier_base_activation] 
    ).to(device)
if model is None:
     logger.error(f"Model selection failed! Model type '{args.model}' not recognized or instantiation failed.")
     sys.exit(1) 

     
logger.info(f"Instantiated model '{model.name}'")
logger.info(f"Starting training process for model: {model.name}")
torch.autograd.set_detect_anomaly(True)
train_model_generic(model,
                    train_loader,
                    test_loader,
                    device,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    gamma=args.gamma,
                    epochs=args.epoch,
                    path=model_save_path)
logger.info(f"Training process completed for model: {model.name}")
