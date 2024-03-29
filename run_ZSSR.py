import GPUtil
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import run_ZSSR_single_input
from PIL import Image
import datetime
from loguru import logger


def main(conf_name, gpu,single_image_path=None):
    if conf_name=='X2_ONE_JUMP_IDEAL_CONF':
        conf = configs.X2_ONE_JUMP_IDEAL_CONF
    elif conf_name== 'X2_IDEAL_WITH_PLOT_CONF':
        conf = configs.X2_IDEAL_WITH_PLOT_CONF
    elif conf_name== 'X2_GRADUAL_IDEAL_CONF':
        conf = configs.X2_GRADUAL_IDEAL_CONF
    elif conf_name== 'X2_GIVEN_KERNEL_CONF':
        conf = configs.X2_GIVEN_KERNEL_CONF
    elif conf_name== 'X2_REAL_CONF':
        conf = configs.X2_REAL_CONF
    else:
        conf = configs.Config()
    logger.info(f"Configuration possible : {conf_name}")
    res_dir = prepare_result_dir(conf)
    local_dir = os.path.dirname(__file__)

    files = [file_path for file_path in glob.glob('%s/*.jpg' % conf.input_path)
             if not file_path[-7:-4] == '_gt']
    
    
    for file_ind, input_file in enumerate(files):
        img = Image.open(input_file)
        img=img.convert('RGB')
        img.save( input_file[:-4]+'.png')

    if single_image_path!=None:
        files=[single_image_path]
    else:
        # We take all png files that are not ground truth
        files = [file_path for file_path in glob.glob('%s/*.png' % conf.input_path)
                if not file_path[-7:-4] == '_gt']
    
    sorties = []
    # Loop over all the files
    for file_ind, input_file in enumerate(files):

        # Ground-truth file needs to be like the input file with _gt (if exists)
        ground_truth_file = input_file[:-4] + '_gt.png'
        if not os.path.isfile(ground_truth_file):
            ground_truth_file = '0'

        # Numeric kernel files need to be like the input file with serial number
        kernel_files = ['%s_%d.mat;' % (input_file[:-4], ind) for ind in range(len(conf.scale_factors))]
        kernel_files_str = ''.join(kernel_files)
        for kernel_file in kernel_files:
            if not os.path.isfile(kernel_file[:-1]):
                kernel_files_str = '0'
                logger.debug('no kernel loaded')
                break
        logger.info(kernel_files)

        # This option uses all the gpu resources efficiently
        if gpu == 'all':

            # Stay stuck in this loop until there is some gpu available with at least half capacity
            gpus = []
            while not gpus:
                gpus = GPUtil.getAvailable(order='memory')

            # Take the gpu with the most free memory
            cur_gpu = gpus[-1]
            # Run ZSSR from command line, open xterm for each run
            
            sortie = run_ZSSR_single_input.main(input_file, ground_truth_file, kernel_files_str, cur_gpu, conf_name, res_dir)

            # Verbose
            logger.debug('Ran file #%d: %s on GPU %d\n' % (file_ind, input_file, cur_gpu))

            # Wait 5 seconds for the previous process to start using GPU. if we wouldn't wait then GPU memory will not
            # yet be taken and all process will start on the same GPU at once and later collapse.
            sleep(5)

        # The other option is just to run sequentially on a chosen GPU.
        else:
            sortie = run_ZSSR_single_input.main(input_file, ground_truth_file, kernel_files_str, gpu, conf_name, res_dir)
        sorties.append(sortie)
    return sorties 

if __name__ == '__main__':
    conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    # Configuration de la journalisation avec Loguru
    
    main(conf_str, gpu_str)
