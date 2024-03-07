import sys
import os
import configs
import ZSSR
from loguru import logger

def main(input_img, ground_truth, kernels, gpu, conf_str, results_path):
    # Choose the wanted GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % gpu

    # 0 input for ground-truth or kernels means None
    ground_truth = None if ground_truth == '0' else ground_truth
    logger.info('*****'), kernels
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # Setup configuration and results directory
    if conf_str=='X2_ONE_JUMP_IDEAL_CONF':
        conf = configs.X2_ONE_JUMP_IDEAL_CONF
    elif conf_str== 'X2_IDEAL_WITH_PLOT_CONF':
        conf = configs.X2_IDEAL_WITH_PLOT_CONF
    elif conf_str== 'X2_GRADUAL_IDEAL_CONF':
        conf = configs.X2_GRADUAL_IDEAL_CONF
    elif conf_str== 'X2_GIVEN_KERNEL_CONF':
        conf = configs.X2_GIVEN_KERNEL_CONF
    elif conf_str== 'X2_REAL_CONF':
        conf = configs.X2_REAL_CONF
    else:
        conf = configs.Config()
    logger.info(f"Configuration choisie : {conf_str}")
    conf.result_path = results_path

    # Run ZSSR on the image
    net = ZSSR.ZSSR(input_img, conf, ground_truth, kernels)
    return net.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
