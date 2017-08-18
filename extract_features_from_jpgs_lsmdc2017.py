import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Model

from keras.preprocessing import image as kerasImage
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jpg_dir = "/hdd4/lsmdc2017/jpgs/"

movie_list = [subdir for subdir in os.listdir(jpg_dir)]

vgg16_model = VGG16(weights='imagenet')
vgg16_layer = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('fc2').output)
inceptionv3_model = InceptionV3(weights='imagenet')
inceptionv3_layer = Model(inputs=inceptionv3_model.input, outputs=inceptionv3_model.get_layer('avg_pool').output)
resnet50_model = ResNet50(weights='imagenet')
resnet50_layer = Model(inputs=resnet50_model.input, outputs=resnet50_model.get_layer('avg_pool').output)

for movie in movie_list:
    clip_list = [clip for clip in os.listdir(os.path.join(jpg_dir, movie))]
    for clip in clip_list:
        print "Processing %s" % clip
        jpg_list = [jpg for jpg in os.listdir(os.path.join(jpg_dir, movie, clip))]
        vgg16_list = []
        inceptionv3_list = []
        resnet50_list = []
        for jpg in sorted(jpg_list):
            jpg_path = os.path.join(jpg_dir, movie, clip, jpg)
            keras_image_224, keras_image_299 = kerasImage.load_img(jpg_path,target_size=(224, 224)), kerasImage.load_img(jpg_path, target_size=(299, 299))
            keras_image_arr_224, keras_image_arr_299 = kerasImage.img_to_array(keras_image_224), kerasImage.img_to_array(keras_image_299)
            keras_image_arr_224, keras_image_arr_299 = np.expand_dims(keras_image_arr_224, axis=0), np.expand_dims(keras_image_arr_299, axis=0)

            vgg16_list.append(vgg16_layer.predict(vgg16_preprocess(keras_image_arr_224)))
            inceptionv3_list.append(inceptionv3_layer.predict(inceptionv3_preprocess(keras_image_arr_299)))
            resnet50_list.append(resnet50_layer.predict(resnet50_preprocess(keras_image_arr_224)))
        np.save(os.path.join(jpg_dir, movie, clip+"_vgg16.npy"), np.array(vgg16_list))
        np.save(os.path.join(jpg_dir, movie, clip+"_inceptionv3.npy"), np.array(inceptionv3_list))
        np.save(os.path.join(jpg_dir, movie, clip+"_resnet50.npy"), np.array(resnet50_list))
