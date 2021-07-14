import os
import sys
import argparse
from PIL import Image
from argparse import ArgumentParser


def ResizeImage(path, width, height):

    if not os.path.exists(path):
        exit('The path is not valid: ' + path)

    img_dst_dir = os.path.join(path, 'resized_images_' + str(width) + '_' + str(height))
    
    print(img_dst_dir)

    if os.path.exists(img_dst_dir):
        exit('The destination path: ' + img_dst_dir + ' already exist. Please rename it')

    os.makedirs(img_dst_dir)

    dirs = os.listdir(path)

    for item in dirs:
        img_path = os.path.join(path, item)
        print(img_path)
        if os.path.isfile(img_path):
            im = Image.open(img_path)
            file_with_ext = os.path.basename(img_path)
            file_no_ext, e = os.path.splitext(file_with_ext)
            imResize = im.resize((width, height), Image.ANTIALIAS)
            img_dst_file_path = os.path.join(img_dst_dir, file_no_ext + '_' + str(width) + '_' + str(height) + '.jpg')
            imResize.save(img_dst_file_path, 'JPEG', quality = 90)


def main():
    parser = argparse.ArgumentParser(description = "Resize images in directory",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dir_path',
        help = 'Path of the directory',
        type = str,
        default = ""
    )

    parser.add_argument(
        '-w', '--width',
        help = 'Image width',
        type = int,
        default = 300
    )
    parser.add_argument(
        '-he', '--height',
        help = 'Image height',
        type = int,
        default = 300
    )

    args = parser.parse_args()
    
    # resizes the images to the desired size
    ResizeImage(args.dir_path, args.width, args.height)


if __name__ == "__main__":
    main()
