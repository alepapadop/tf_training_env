import os
import subprocess
import sys
import argparse
from argparse import ArgumentParser

PRE_ANNOTATED_IMAGES_PATH = 'workspace/training_demo/images'

current_working_model_path = ""

model_pipeline_dir_name = ""

def CreateFolderStructure():
    global current_working_model_path
    global model_pipeline_dir_name

    dst_dir = os.path.join('workspace')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'annotations')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'exported_models')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'images', 'test')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'images', 'train')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'models')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('workspace', 'training_demo', 'pre_trained_models')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_dir = os.path.join('scripts', 'preprocessing')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    loop = True
    if model_pipeline_dir_name:
        loop = False
        dst_dir = os.path.join('workspace', 'training_demo', 'models', model_pipeline_dir_name)
        current_working_model_path = dst_dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)


    count = 1
    while loop:
        dst_dir = os.path.join('workspace', 'training_demo', 'models', 'my_model_' + str(count))
        count = count + 1;
        if not os.path.exists(dst_dir):
            current_working_model_path = dst_dir
            os.makedirs(dst_dir)
            break

def IsDirEmpty(dir_name):
    
    is_empty = False

    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            #"Directory is empty"
            is_empty = True
        else:
            #"Directory is not empty"
            is_empty = False
    else:
        #Given directory doesn't exist
        is_empty = True
    
    return is_empty 


def PartitionImages():
    script_src = os.path.join('scripts', 'preprocessing', 'partition_dataset.py')
    
    image_dir = os.path.join('workspace', 'training_demo', 'images')
    
    train_dir = os.path.join('workspace', 'training_demo', 'images', 'train')
    test_dir = os.path.join('workspace', 'training_demo', 'images', 'test')

    print('Pleae place the resized images inside the: ' + image_dir)
    input('Press ENTER to continue ...')
    
    if IsDirEmpty(train_dir) and IsDirEmpty(test_dir):

        ret = subprocess.check_call([sys.executable, script_src, '-i', image_dir, '-o', image_dir, '-r', '0.2', '-x'])

        if ret != 0:
            exit('Image Partition script failed. Check the partision_dataset.py inside script/preprocessing/')
    else:
        print('There are already images in the train and test direcories. No partition is made')


def CreateLabelMapAndRecordFile():

    label_map_file = os.path.join('workspace', 'training_demo', 'annotations', 'label_map.pbtxt')
    if not os.path.exists(label_map_file):
        with open(label_map_file, 'w'): pass

    filesize = os.path.getsize(label_map_file)
    if filesize == 0:
        print('A label_map.pbtxt file is created. Please go to workspace/training_demo/annotations and fill in the class id and class name for the images')
        print('After finishing press ENTER to continue. If you press  enter by mistake just run again the whole script. No data will be losts.')

        input("Press Enter to continue...")
    
    filesize = os.path.getsize(label_map_file)
    if filesize == 0:
        exit('The label_map.pbtxt is empty. Please fill in the class data in the file and re-run the whole script. No data will be lost')
    
    script_src = os.path.join('scripts', 'preprocessing', 'generate_tfrecord.py')

    train_dir = os.path.join('workspace', 'training_demo', 'images', 'train')
    test_dir = os.path.join('workspace', 'training_demo', 'images', 'test')
    train_record_file = os.path.join('workspace', 'training_demo', 'annotations', 'train.record')
    test_record_file = os.path.join('workspace', 'training_demo', 'annotations', 'test.record')

    ret = subprocess.check_call([sys.executable, script_src, '-x', train_dir, '-l', label_map_file, '-o', train_record_file])

    if ret != 0:
        exit('Image TFRocord generation for the train image set failed.')

    ret = subprocess.check_call([sys.executable, script_src, '-x', test_dir, '-l', label_map_file, '-o', test_record_file])

    if ret != 0:
        exit('Image TFRocord generation for the test image set failed.')

def CheckBeforeModelTrain():
    #check for pipeline file and chk
    
    if not current_working_model_path or not os.path.exists(current_working_model_path):
        exit('Current working model path not found. Searching for: ' + current_working_model_path)
    
    pipeline_config_path = os.path.join(current_working_model_path, 'pipeline.config') 

    if not os.path.exists(pipeline_config_path):
        print('The pipeline.config file is missing in path: ' + current_working_model_path)
        print('Print please create a file or copy one from an existin pre-trained model')
        print('After finishing pres ENTER to continue. If you press  enter by mistake just run again the whole script. No data will be losts.')

        input('Press Enter to continue...')

    if not os.path.exists(pipeline_config_path):
        exit('The pipeline.config file ismissing. Please cretate a file and press re-run the whole script. No data will be lost')
    

def main():
    global model_pipeline_dir_name

    # Initiate argument parser
    parser = argparse.ArgumentParser(description = "Prepare the data for training",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-p', '--pipeline_config_dir_name',
        help = 'Name of an existing directory for searching the pipeline.config file. If not set a new directory will be created and ask to add a config file.',
        type = str,
        default = ""
    )

    args = parser.parse_args()
    
    model_pipeline_dir_name = args.pipeline_config_dir_name

    CreateFolderStructure()
    PartitionImages()
    CreateLabelMapAndRecordFile()
    CheckBeforeModelTrain()



if  __name__ == "__main__":
    main()
