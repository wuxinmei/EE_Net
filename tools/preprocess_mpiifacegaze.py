#!/usr/bin/env python

import pathlib
import h5py
import numpy as np
import tqdm, argparse
from PIL import Image
import random
import configparser
import os
from random import randint
import shutil


def add_mat_data_to_hdf5(person_id: str, dataset_dir: pathlib.Path,
                         output_path: pathlib.Path) -> None:
    with h5py.File(dataset_dir / f'{person_id}.mat', 'r') as f_input:
        images = f_input.get('Data/data')[()]
        labels = f_input.get('Data/label')[()][:, :4]
        Leye = f_input.get('Data/label')[()][:, 4:8]
        Leye = abs(Leye)
        Reye = f_input.get('Data/label')[()][:, 8:12]
    assert len(images) == len(labels) == 3000

    images = images.transpose(0, 2, 3, 1).astype(np.uint8)
    poses = labels[:, 2:]
    gazes = labels[:, :2]
    LeyeCs = (Leye[:, :2]+Leye[:, 2:])/2
    ReyeCs = (Reye[:, :2]+Reye[:, 2:])/2

    Leye_sizes = 1.7*(LeyeCs[:, 0]-Leye[:, 0])
    Reye_sizes = 1.7*(ReyeCs[:, 0]-Reye[:, 0])

    leftEyeBbox = np.stack((LeyeCs[:, 0]-Leye_sizes, LeyeCs[:, 1]-Leye_sizes, 2*Leye_sizes, 2*Leye_sizes), axis=1).astype(int)
    rightEyeBbox = np.stack((ReyeCs[:, 0]-Reye_sizes, ReyeCs[:, 1]-Reye_sizes, 2*Reye_sizes, 2*Reye_sizes), axis=1).astype(int)

    output = './datasets/test'
    FacePath = preparePath(os.path.join(output, f'{person_id}/FaceAug'))
    # with h5py.File(output_path, 'a') as f_output:
    for index, (image, gaze,
                pose) in tqdm.tqdm(enumerate(zip(images, gazes, poses)),
                                   leave=False):
        cf = configparser.ConfigParser()
        cf.read('config.ini', encoding='utf-8')
        png = cf.get('config', 'png')
        image = np.asarray(paste_enhance(png, image))
        # Crop images
        # imEyeL = cropImage(image, leftEyeBbox[index, :])
        # imEyeR = cropImage(image, rightEyeBbox[index, :])
        image = image[:, :, ::-1]

        Image.fromarray(image, mode='RGB').save(os.path.join(FacePath, f'{index:04}.jpg'), quality=100)

        # groundtruth = open(os.path.join(FacePath, f'groundtruth.txt'), 'a')
        # groundtruth.writelines('index: %4d\t gaze vector: %s\t pose vector: %s\n' % (index, gaze, pose))

            # # Save images to h5
            # f_output.create_dataset(f'{person_id}/imEyeL/{index:04}', data=imEyeL)
            # f_output.create_dataset(f'{person_id}/imEyeR/{index:04}', data=imEyeR)
            # f_output.create_dataset(f'{person_id}/image/{index:04}', data=image)
            # f_output.create_dataset(f'{person_id}/pose/{index:04}', data=pose)
            # f_output.create_dataset(f'{person_id}/gaze/{index:04}', data=gaze)


def Augmentdata(person_id: str, index: int, dataset_dir: pathlib.Path,
                         output_path: pathlib.Path) -> None:
    with h5py.File(dataset_dir, 'r') as f_input:
        imEyeL = f_input.get(f'{person_id}/imEyeL/{index:04}')[()]
        imEyeR = f_input.get(f'{person_id}/imEyeR/{index:04}')[()]
        image = f_input.get(f'{person_id}/image/{index:04}')[()]
        pose = f_input.get(f'{person_id}/pose/{index:04}')[()]
        gaze = f_input.get(f'{person_id}/gaze/{index:04}')[()]

    with h5py.File(output_path, 'a') as f:
        cf = configparser.ConfigParser()
        cf.read('config.ini', encoding='utf-8')
        png = cf.get('config', 'png')
        FacePath = preparePath(os.path.join(output_path, f'{person_id}/Face'))
        # for index in tqdm.tqdm(range(3000)):
        # if person_id in ['p00', 'p01', 'p02', 'p03', 'p04', 'p05']:
        if index <= 1499:
            imEyeL = np.asarray(paste_enhance(png, imEyeL))
            # imEyeR = paste_enhance(png, imEyeR)
            # Image.fromarray(imEyeL).save(os.path.join(FacePath, f'{index:04}.jpg'), quality=95)
            # Image.fromarray(imEyeR).save(os.path.join(FacePath, f'{index:04}.jpg'), quality=95)
            f.create_dataset(f'{person_id}/imEyeL/{index+3000:04}', data=imEyeL)
            f.create_dataset(f'{person_id}/imEyeR/{index+3000:04}', data=imEyeR)
            f.create_dataset(f'{person_id}/image/{index+3000:04}', data=image)
            f.create_dataset(f'{person_id}/pose/{index+3000:04}', data=pose)
            f.create_dataset(f'{person_id}/gaze/{index+3000:04}', data=gaze)
        else:
            # imEyeL = paste_enhance(png, imEyeL)
            imEyeR = paste_enhance(png, imEyeR)
            # Image.fromarray(imEyeR).save('%05d R.jpg' % index, quality=95)
            # Image.fromarray(imEyeL).save('%05d L.jpg' % index, quality=95)
            f.create_dataset(f'{person_id}/imEyeL/{index+3000:04}', data=imEyeL)
            f.create_dataset(f'{person_id}/imEyeR/{index+3000:04}', data=imEyeR)
            f.create_dataset(f'{person_id}/image/{index+3000:04}', data=image)
            f.create_dataset(f'{person_id}/pose/{index+3000:04}', data=pose)
            f.create_dataset(f'{person_id}/gaze/{index+3000:04}', data=gaze)

def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
    res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]
    return res

def paste_enhance(png_path, main_img):
    # os.makedirs(save_images_path, exist_ok=True)
    files = os.listdir(png_path)
    # paste_files = os.listdir(images_paste_path)

    total_num = len(files)
    idx = randint(0, total_num)
    idx = idx % total_num
    image_path = files[idx]
    picture_path = os.path.join(os.path.abspath(png_path), image_path)

    im = Image.open(picture_path).convert('RGBA')  # 读取蒙层

    paste_num = 1
    for n in range(paste_num):  # 蒙层在主图上粘贴的次数，默认一次
        main_img = Image.fromarray(np.uint8(main_img)).convert('RGBA')

        w, h = main_img.size
        max = (w if (w < h) else h)

        random_result = dict()
        # random_result['num_resize'] = random.randint(30, 40)  # 闭区间,保存随机resize尺寸
        random_result['num_resize'] = int(max*0.4)
        random_result['rotate_F'] = random.randint(1, 30)  # 随机逆时针旋转角度、随机顺时针旋转角度
        random_result['rotate_T'] = random.randint(-30, -1)  # 随机顺时针旋转角度
        random_result['status'] = random.randint(-1, 0)  # 随机抽状态，-1逆时针，0顺时针

        # print(random_result['num_resize'], main_img.size)
        try:
            random_result['cut_x_left'] = random.randint(0, w - random_result['num_resize'])  # 随机主图-蒙层图范围的左上角x
            random_result['cut_y_left'] = random.randint(0, h - random_result['num_resize'])  # 随机主图-蒙层图范围的左上角y
        except ValueError:  # 蒙层图resize比主图尺寸大,那就不粘贴
            continue

        img_resize = im.resize((random_result['num_resize'], random_result['num_resize']))  # resize图片

        if random_result['status'] == -1:  # 随机旋转图片
            img_rotate = img_resize.rotate(random_result['rotate_F'])
        else:
            img_rotate = img_resize.rotate(random_result['rotate_T'])

        x_left, y_left = random_result['cut_x_left'], random_result['cut_y_left']
        x_right = random_result['cut_x_left'] + random_result['num_resize']
        y_right = random_result['cut_y_left'] + random_result['num_resize']

        main_img.paste(img_rotate, (x_left, y_left, x_right, y_right), img_rotate)  # 粘贴蒙层图片
        save_img = main_img.convert('RGB')  # 转换为jpg图片

        return save_img
def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/media/echo/2016-2020/Cartolab/eye gaze/dataset/MPIIFaceGaze/MPIIFaceGaze_normalizad', type=str, required=False)
    parser.add_argument('--output-dir', '-o', default='./datasets/test', type=str, required=False)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'MPIIFaceGazeFull.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)
    for person_id in tqdm.tqdm(range(15)):
        person_id = f'p{person_id:02}'
        add_mat_data_to_hdf5(person_id, dataset_dir, output_path)

    # aug_dataset_dir = output_dir / 'MPIIFaceGazeFull.h5'
    # for person_id in tqdm.tqdm(range(15)):
    #     person_id = f'p{person_id:02}'
    #     print('aug person id', person_id)
    #     for index in tqdm.tqdm(range(3000)):
    #         # print("aug index:", index)
    #         Augmentdata(person_id, index, output_path, aug_dataset_dir)


if __name__ == '__main__':
    main()


# #!/usr/bin/env python
#
# import pathlib
# import shutil
#
# import h5py
# import numpy as np
# import tqdm, os, argparse
# from PIL import Image
#
#
#
# def add_mat_data_to_hdf5(person_id: str, dataset_dir: pathlib.Path, output_dir: pathlib.Path,
#                          output_path: pathlib.Path) -> None:
#     with h5py.File(dataset_dir / f'{person_id}.mat', 'r') as f_input:
#         images = f_input.get('Data/data')[()]
#         labels = f_input.get('Data/label')[()][:, :4]
#         Leye = f_input.get('Data/label')[()][:, 4:8]
#         Leye = abs(Leye)
#         Reye = f_input.get('Data/label')[()][:, 8:12]
#     assert len(images) == len(labels) == 3000
#
#     images = images.transpose(0, 2, 3, 1).astype(np.uint8)
#     # images = images.astype(np.uint8)
#     poses = labels[:, 2:]
#     gazes = labels[:, :2]
#     LeyeCs = (Leye[:, :2]+Leye[:, 2:])/2
#     ReyeCs = (Reye[:, :2]+Reye[:, 2:])/2
#
#     Leye_sizes = 1.7*(LeyeCs[:, 0]-Leye[:, 0])
#     Reye_sizes = 1.7*(ReyeCs[:, 0]-Reye[:, 0])
#
#     leftEyeBbox = np.stack((LeyeCs[:, 0]-Leye_sizes, LeyeCs[:, 1]-Leye_sizes, 2*Leye_sizes, 2*Leye_sizes), axis=1).astype(int)
#     rightEyeBbox = np.stack((ReyeCs[:, 0]-Reye_sizes, ReyeCs[:, 1]-Reye_sizes, 2*Reye_sizes, 2*Reye_sizes), axis=1).astype(int)
#
#
#
#     with h5py.File(output_path, 'a') as f_output:
#         for index, (image, gaze,
#                     pose) in tqdm.tqdm(enumerate(zip(images, gazes, poses)),
#                                        leave=False):
#
#             # Crop images
#             imEyeL = cropImage(image, leftEyeBbox[index, :])
#             imEyeR = cropImage(image, rightEyeBbox[index, :])
#
#
#             # # Save images
#
#             Image.fromarray(image).save(os.path.join(FacePath, f'{index:04}.jpg'), quality=95)
#             # Image.fromarray(imEyeL).save(os.path.join(args.output_dir, '%05d L.jpg' % j), quality=95)
#             # Image.fromarray(imEyeR).save(os.path.join(args.output_dir, '%05d R.jpg' % j), quality=95)
#             # f_output.create_dataset(f'{person_id}/imEyeL/{index:04}', data=imEyeL)
#             # f_output.create_dataset(f'{person_id}/imEyeR/{index:04}', data=imEyeR)
#             # f_output.create_dataset(f'{person_id}/image/{index:04}', data=image)
#             # f_output.create_dataset(f'{person_id}/pose/{index:04}', data=pose)
#             # f_output.create_dataset(f'{person_id}/gaze/{index:04}', data=gaze)
#
# def cropImage(img, bbox):
#     bbox = np.array(bbox, int)
#
#     aSrc = np.maximum(bbox[:2], 0)
#     bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))
#
#     aDst = aSrc - bbox[:2]
#     bDst = aDst + (bSrc - aSrc)
#
#     res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
#     res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]
#     return res
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', default='/media/ubuntu/2016-20201/Cartolab/eye_gaze_tracking/dataset/MPIIGaze/MPIIFaceGaze_normalizad', type=str, required=False)
#     parser.add_argument('--output-dir', '-o', default='datasets/test/', type=str, required=False)
#     args = parser.parse_args()
#
#     output_dir = pathlib.Path(args.output_dir)
#     output_dir.mkdir(exist_ok=True, parents=True)
#     output_path = output_dir / 'MPIIFaceGaze.h5'
#     # if output_path.exists():
#     #     raise ValueError(f'{output_path} already exists.')
#
#     dataset_dir = pathlib.Path(args.dataset)
#     for person_id in tqdm.tqdm(range(15)):
#         person_id = f'p{person_id:02}'
#
#         add_mat_data_to_hdf5(person_id, dataset_dir, output_dir, output_path)
#
#
# if __name__ == '__main__':
#     main()

