import os
import subprocess
import sys
import requests
import glob
import tempfile
from zipfile import ZipFile
from git import Repo, RemoteProgress
from tqdm import tqdm
from shutil import copyfile
from argparse import ArgumentParser

URL_GIT_TENSORFLOW_MODEL = 'https://github.com/tensorflow/models.git'
PB_REL = 'https://github.com/protocolbuffers/protobuf/releases'

PB_LAST_VERSION = '/download/v3.17.0/protoc-3.17.0-linux-x86_64.zip'



class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()

def CloneModelsRepo():
    work_dir = os.getcwd()
    dest_dir = os.path.join(work_dir, 'models')
    Repo.clone_from(URL_GIT_TENSORFLOW_MODEL, dest_dir, progress = CloneProgress())

def InstallProtoBuffer():

    dst_file = os.path.join('models', 'research', 'protoc_buffers.zip')
    
    r = requests.get(PB_REL + PB_LAST_VERSION, allow_redirects = True)

    total = int(r.headers.get('content-length', 0))
    print('Downloading: ' + PB_REL + PB_LAST_VERSION)
    with open(dst_file, 'wb') as file, tqdm (
        desc = 'Progress',
        total = total,
        unit = 'iB',
        unit_scale = True,
        unit_divisor = 1024,
        ) as bar:
        for data in r.iter_content(chunk_size = 1024):
            size = file.write(data)
            bar.update(size)
    
    dst_dir = os.path.join('models', 'research', 'proto_buffers')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with ZipFile(dst_file, 'w') as zipObj:
        zipObj.extractall(dst_dir)

    protoc_exec = os.path.join('proto_buffers', 'bin', 'protoc')
    proto_files = os.path.join('object_detection', 'protos')
    python_out = os.path.join('models', 'research')
    
    wd_new = os.path.join('models', 'research')
    wd_back = os.getcwd()
    os.chdir(wd_new)

    os.chmod(protoc_exec, 0o544)
    
    files = glob.glob(proto_files + os.path.sep + '*.proto')

    for f in files:
        subprocess.check_call([protoc_exec, f, '--python_out=.'])

    os.chdir(wd_back)

def InstallObjectDetectionApi():
    src_file = os.path.join('models', 'research', 'object_detection', 'packages', 'tf2', 'setup.py')
    dst_file = os.path.join('models', 'research', 'setup.py')

    ist_path = os.path.join('models', 'research')

    copyfile(src_file, dst_file)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', ist_path])

def VerifyTensorflowInstallation():

    cmd = [sys.executable, '-c', '"import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"']
    #proc = subprocess.check_call(cmd)
    # find a way to print the output
    #print(proc)

    #cmd = [sys.executable, '--version']
    #with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
    #    sys.stdout.buffer.write(proc.stdout.read())
    #    print(proc.stdout.read())

def VerifyInstallationComplete():
    path = os.path.join('models', 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    cmd = [sys.executable, path]
    proc = subprocess.check_call(cmd)
    print('')

def main():
    #VerifyTensorflowInstallation()
    #CloneModelRepo()
    #InstallProtoBuffer()
    #InstallObjectDetectionApi()
    #VerifyInstallationComplete()

if __name__ == "__main__":
    main()

