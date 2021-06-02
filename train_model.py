import os
import subprocess
import sys
import argparse
from argparse import ArgumentParser
from shutil import copyfile


def CopyModelTrainer():
    src_file = os.path.join('models', 'research', 'object_detection', 'model_main_tf2.py')
    dst_file = os.path.join('models', 'research', 'model_main_tf2.py')

    if os.path.exists(dst_file):
        print('Training script file: ' + dst_file + ' already exists. Remove or rename to copy from the object detection source dir')
    else:
        copyfile(src_file, dst_file)



def TrainModel(model_dir_name):
    model_dir = os.path.join('workspace', 'training_demo', 'models', model_dir_name)

    if not os.path.exists(model_dir):
        exit('The directory: ' + model_dir + 'does not exists')

    train_script_path = os.path.join('model_main_tf2.py')
    model_dir_path = os.path.join('models', model_dir_name)
    pipeline_path_dir = os.path.join('models', model_dir_name, 'pipeline.config')

    cwd = os.getcwd()
    new_cwd = os.path.join('workspace', 'training_demo')
    os.chdir(new_cwd)

    #python3 model_main_tf2.py --model_dir=models/my_model --pipeline_config_path=models/my_model/pipeline.config
    # do not put the = in the arguments when calling from a python script
    ret = subprocess.check_call([sys.executable, train_script_path, '--model_dir', model_dir_path, '--pipeline_config_path', pipeline_path_dir])
    

    if ret != 0:
        exit('Training script failed')

    os.chdir(cwd)

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description = "Train the model",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model_dir_name',
        help = 'Name of the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type = str,
        default = ""
    )

    args = parser.parse_args()

    if not args.model_dir_name:
        exit('No model directory name provided. Use -m or --model_dir for the directory')
    
    CopyModelTrainer()
    TrainModel(args.model_dir_name)


if __name__ == "__main__":
    main()
