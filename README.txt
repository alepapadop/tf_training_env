1) Create a python virtual env

	python3 -m venv ./tf

2) Activate the virtual env

	source ./tf/bin/activate  # sh, bash, or zsh

3) Upgrade pip
	
	pip install --upgrade pip

4) install tensorflow
	
	pip install six

	pip install appdirs

	pip install packaging

	pip install ordered_set

	pip install pycurl

	pip install --upgrade tensorflow
	
	pip install gitpython

	pip install tqdm

	pip install labelImg

	pip install pandas

	pip install tflite-support

5) Tests installation
	
	python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

6) Install the Google Proto Buffer
	
	You can set the protoc executable in any directory, the following command 
	works if the protoc is set in the PATH	

	protoc object_detection/protos/*.proto --python_out=.

7) Install the object detection api
	
	From within TensorFlow/models/research/

	cp models/research/object_detection/packages/tf2/setup.py .
	python3 -m pip install .

8) Test the installation
	
	From within TensorFlow/models/research/

	python object_detection/builders/model_builder_tf2_test.py

9) Create the file structure for training

	TensorFlow/
	├─ addons/ (Optional)
	│  └─ labelImg/
	├─ models/
	│  ├─ community/
	│  ├─ official/
	│  ├─ orbit/
	│  ├─ research/
	│  └─ ...
	└─ workspace/
	   └─ training_demo/
		├─ annotations/
		├─ exported_models/
		├─ images/
		│	├─ test/
		│	└─ train/
		├─ models/
		└─ pre_trained_models

10) Create a annotated image set or download a pre-annotated image set, coco image annotation format must be used

11) Move the images with the annotations inside the workspace/training_demo/images/ and place a 80% in the train and 20% in test

12) create the label_map.pbtxt placed inside training_demo/annotations

item {
	id:1
	name: 'pencil'
}

etc

13) Create Tensorflow Records
	
	For example run
		
	python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

14) Download a pre-trained model

15) Get the pipeline.config from the pretrained model and edit it for your model

16) Copy the /models/research/object_detection/model_main_tf2.py into thr training_demo direcotry and run

	python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

17) Evaluate the model
	
	python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn

18) Monitor the jobs

	tensorboard --logdir=models/my_ssd_resnet50_v1_fpn

19) Export a trained model
	
	python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_efficientdet_d1\pipeline.config --trained_checkpoint_dir .\models\my_efficientdet_d1\ --output_directory .\exported-models\my_model

20) Check your trained model
	

###############################################################################################################

The simple way:

1) Run "bootstrap.py" to install the requires packages only one time is needed

2) Run the resize_images.py to resize your images for training

	default size 300*300 

	use the -d "direcorty" to specify where the images are located

	use the -w -he for specifing the width and the height of the image
	
	a new directory will be created named resized_images_width_height

3) Run "prepare_workspace.py" to create the required folder structure only one time is needed
	
	use the the -p "model_name" to create a direcotry with a specific name inside the 
	
	workspace/training_demo/models/

	the images from the previous step must be placed inside wordspace/training_demo/images

	the images will be partitioned in workspace/training_demo/images/train and workspace/training_demo/images/test

	a label_map.pbtxt file is created inside workspace/training_demo/annotations/

	the user must fill inside the data

	two TFRecord files will be generated the test.record and train.record

	The pipeline.config must be placed by the user inside the workspace/training_demo/models/my_model
	
	the suer must edit the pipeline.config.

	A test runs to check the installation

4) Run train_model.py to start the training

	use -m the model direcotory name to selecet the correct model to train

	the training data are placed inside the workspace/training_demo/models/my_model

5) Run export_model.py to export the model
	
	use -p for the model name to selcet the correct model

	the exported model is placed in workspace/training_demo/exported_models/my_model

6) Use im_detections.py to test the model

	use -n to specify the name of the exported model

	use -d to specify the direcorty with the resized images

	use -t to specify the threshold of the detection, default 0.3

7) Use the tf_lite_converter.py to convert the model to tf lite
	
	use -m to specify the nameo of the exported model
