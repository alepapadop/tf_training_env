import tensorflow as tf
import sys
import os
import argparse
from argparse import ArgumentParser


def ConvertModel(exported_model_name):
    
    model_dir = os.path.join('workspace', 'training_demo', 'exported_models', exported_model_name, 'saved_model')

    lite_model_path = os.path.join(model_dir, 'model.tflite')

    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

    converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
            ]


    tflite_model = converter.convert()

    with open(lite_model_path, 'wb') as f:
       f.write(tflite_model)




def main():
    parser = argparse.ArgumentParser(description = "Test the model on image",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--exported_model_name',
        help = 'Name of an existing directory containing the trained model.',
        type = str,
        default = ""
    )

    args = parser.parse_args()

    if not args.exported_model_name:
        exit('Please provide the model name')

    ConvertModel(args.exported_model_name)

if __name__ == "__main__":
    main()
