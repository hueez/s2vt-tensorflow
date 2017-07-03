import numpy as np
import os
import skimage
import cv2
import tensorflow as tf
import pandas as pd
import re
import random
from keras.applications.vgg16 import VGG16 as vgg16
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from tensorflow import app

# os.environ["CUDA_VISIBLE_DEVICES"] = 0,1,2,3,4,5

class Vocabulary(object):
    def __init__(self, csv="/home/jaehui/data/msvd/msvd.csv", word_count_threshold=4):
        self.df = pd.read_csv(csv, header=0)
        self.wordtoix, self.ixtoword, self.bias_init_vector = self.create_vocab(word_count_threshold)

    def get_desc_vec_by_id(self, video_id):
        descriptions = self.df.loc[
            (self.df['VideoID'] == video_id) & (self.df['Language'] == "English"), ["Description"]]
        desc_list = descriptions["Description"].tolist()

        vectors = []
        for sentence in desc_list:
            sentence = re.sub("[^a-zA-Z0-9 ]+", "", str(sentence))
            vector = []
            for word in sentence.split(" "):
                word = word.lower()
                if word in self.wordtoix:
                    vector.append(self.wordtoix[word])
                else:
                    vector.append(self.wordtoix['<unk>'])

        return vectors

    def get_desc_by_id(self, video_id):
        descriptions = self.df.loc[
            (self.df['VideoID'] == video_id) & (self.df['Language'] == "English"), ["Description"]]
        desc_list = descriptions["Description"].tolist()

        return desc_list

    def create_vocab(self, word_count_threshold):
        word_counter = {}
        nsents = 0

        ret = self.df.loc[self.df['Language'] == "English", ["Description"]]
        desc_list = ret["Description"].tolist()
        f = open("outputs", "w")
        desc_list = [re.sub("[^a-zA-Z0-9 ]+", "", str(x)) for x in desc_list]

        for sentence in desc_list:
            nsents += 1
            for word in sentence.split(' '):
                word = word.lower()
                if len(word) is not 0:
                    word_counter[word] = word_counter.get(word, 0) + 1

        vocab = [word for word in word_counter if word_counter[word] >= word_count_threshold]

        ixtoword = {}
        ixtoword[0] = '<pad>'
        ixtoword[1] = '<bos>'
        ixtoword[2] = '<eos>'
        ixtoword[3] = '<unk>'

        wordtoix = {}
        wordtoix['<pad>'] = 0
        wordtoix['<bos>'] = 1
        wordtoix['<eos>'] = 2
        wordtoix['<unk>'] = 3

        for idx, w in enumerate(vocab):
            wordtoix[w] = idx + 4
            ixtoword[idx + 4] = w

        word_counter['<pad>'] = nsents
        word_counter['<bos>'] = nsents
        word_counter['<eos>'] = nsents
        word_counter['<unk>'] = nsents

        bias_init_vector = np.array([1.0 * word_counter[ixtoword[i]] for i in ixtoword])
        bias_init_vector /= np.sum(bias_init_vector)
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector)

        return wordtoix, ixtoword, bias_init_vector


class VideoFeatureExtractor(object):
    def __init__(self):
        pretrained = vgg16(weights='imagenet')
        self.model = Model(inputs=pretrained.inputs, outputs=pretrained.get_layer('fc2').output)

    def convert_to_feature_list(self, frame_list):
        feature_list = []
        for frame in frame_list:
            frame = resize_and_crop_image(frame)
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)
            feat = self.model.predict(frame)
            feature_list.append(feat.tolist()[0])
        return feature_list

    def extract_raw_frames(self, video_file, n_frames=80, is_randomly_spaced=True):
        try:
            cap = cv2.VideoCapture(video_file)
        except:
            pass

        frame_list = []

        while True:
            ret, frame = cap.read()

            if ret is False:
                break
            frame_list.append(frame)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_list = np.array(frame_list)

        if frame_count > n_frames:
            if is_randomly_spaced:
                frame_indices = sorted(random.sample(range(0, frame_count), n_frames))
            else:
                frame_indices = np.linspace(0, frame_count, num=n_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]
        return frame_list

# borrowed from im2txt
def int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def float_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def float_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def resize_and_crop_image(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.title(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape

    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]
    return cv2.resize(resized_image, (target_height, target_width))


def print_record(filename, ixtoword={}):
    for example in tf.python_io.tf_record_iterator(filename):

        tf_seq_example = tf.train.SequenceExample.FromString(example)

        video_id = tf_seq_example.context.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
        desc_id = tf_seq_example.context.feature["description_id"].int64_list.value[0]
        description = tf_seq_example.context.feature['description'].int64_list.value

        print "video id: ", video_id
        print "desc id: ", desc_id
        if ixtoword is None:
            print "description: ", description
        else:
            print "description: ",
            for ix in description:
                print ixtoword[int(ix)],
            print ""

        vgg_fc7 = []
        n_frames = len(tf_seq_example.feature_lists.feature_list['vgg16_fc7'].feature)
        for f in range(n_frames):
            frame = tf_seq_example.feature_lists.feature_list["vgg16_fc7"].feature[f].float_list.value
            print "frameSeq: ", f
            listed = []
            for i, _ in enumerate(frame):
                listed.append(frame[i])
            print listed
            vgg_fc7.append(listed)
        print "frame feature shape: ", np.array(vgg_fc7).shape


def build(video_dir, output_dir, extractor, voc):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = os.listdir(video_dir)
    video_files = filter(lambda x: x.endswith('mp4'), video_files)

    for idx, video_file in enumerate(video_files):
        video_id = video_file.split("_")[0]
        video_file_path = os.path.join(video_dir, video_file)
        description_list = voc.get_desc_vec_by_id(video_id)

        for desc_id, desc in enumerate(description_list):

            features_from_randomly_spaced_frames = extractor.extract_raw_frames(video_file_path)
            # features_from_evenly_spaced_frames = extractor.extract_raw_frames(video_file_path, is_randomly_spaced=False)
            vgg_feature_list = extractor.convert_to_feature_list(features_from_randomly_spaced_frames)

            with tf.python_io.TFRecordWriter(
                    os.path.join(output_dir, video_file + '_' + str(desc_id) + '.tfrecord')) as writer:

                context = tf.train.Features(feature={
                    "video_id": bytes_feature(video_id),
                    "description_id": int64_feature(desc_id)
                })

                for word_id in desc:
                    context.feature["description"].int64_list.value.append(word_id)

                feature_lists = tf.train.FeatureLists(feature_list={
                    "vgg16_fc7": float_feature_list(vgg_feature_list)
                })

                sequence_example = tf.train.SequenceExample(
                    context=context, feature_lists=feature_lists
                )
                print ('Writing frame features to ' + video_file + '_' + str(desc_id) + '.tfrecord')
                writer.write(sequence_example.SerializeToString())

def main(unused_argv):

    keras_vgg = VideoFeatureExtractor()
    voc = Vocabulary(word_count_threshold=0)

    '''
    feat = VideoFeatureExtractor()
    frame_list = feat.extract_raw_frames("/home/jaehui/data/msvd/YouTubeClips/NjCqtzZ3OtU_62_67.mp4")
    feature_list = feat.convert_to_feature_list(frame_list)


    for feature in feature_list:
        print len(feature)

    print len(feature_list)

    '''

    build(video_dir="/home/jaehui/data/msvd/YouTubeClips", output_dir="/home/jaehui/data/msvd/records", extractor=keras_vgg, voc=voc)
    #print_record("/home/jaehui/data/NjCqtzZ3OtU_62_67.mp4_0.tfrecord", voc.ixtoword)

if __name__ == "__main__":
    app.run()
