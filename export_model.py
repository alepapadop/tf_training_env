import os
import subprocess
import sys
import requests
import glob
import tempfile
import argparse
from zipfile import ZipFile
from git import Repo, RemoteProgress
from tqdm import tqdm
from shutil import copyfile
from argparse import ArgumentParser

def CopyModelExporter():
    src_file = os.path.join('models', 'research', 'object_detection', 'exporter_main_v2.py')
    dst_file = os.path.join('models', 'research', 'exporter_main_v2.py')
    
    if os.path.exists(dst_file):
        print('Exporter script file: ' + dst_file + ' already exists. Remove or rename to copy from the object detection source dir')
    else:
        copyfile(src_file, dst_file)

def RunModelExporter(model_dir_name):
    model_dir = os.path.join('workspace', 'training_demo', 'models', model_dir_name)

    if not os.path.exists(model_dir):
        exit('The directory: ' + model_dir + 'does not exists')

    script_file = os.path.join('exporter_main_v2.py')

    pipeline_config = os.path.join('models', model_dir_name, 'pipeline.config')

    checkpoint = os.path.join('models', model_dir_name)
    output = os.path.join('exported_modesl', model_dir_name)

    cwd = os.getcwd() 
    new_cwd = os.path.join('workspace', 'training_demo')
    os.chdir(new_cwd)

    #python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_model/pipeline.config --trained_checkpoint_dir models/my_model/ --output_directory exported_models/my_model
    ret = subprocess.check_call([sys.executable, script_file, '--input_type', 'image_tensor', '--pipeline_config_path', pipeline_config, '--trained_checkpoint_dir', checkpoint, '--output_directory', output])
    
    os.chdir(cwd)

    if ret != 0:
        exit('Training script failed')



def main():

    parser = argparse.ArgumentParser(description = "Export the model",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-p', '--model_dir_name',
        help = 'Name of an existing directory containing the trained model.',
        type = str,
        default = ""
    )

    args = parser.parse_args()

    if not args.model_dir_name:
        exit('Please provide the model name')

    CopyModelExporter()
    RunModelExporter(args.model_dir_name)



if __name__ == "__main__":
    main()
