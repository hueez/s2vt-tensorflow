from collections import Counter
from collections import namedtuple
from datetime import datetime
import os.path
import random
import sys
import threading
import cv2
import pandas as pd

import nltk.tokenize
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.flags.DEFINE_string("data_dir", "/hdd4/lsmdc2017", "data directory")
'''
tf.flags.DEFINE_string("train_file", "LSMDC16_annos_training.csv",
                       "Annotations for training video files")
tf.flags.DEFINE_string("val_file", "LSMDC16_annos_val.csv",
                       "Annotations for validating video files")
'''
tf.flags.DEFINE_string("train_file", "small_training.csv",
                       "Annotations for training video files")
tf.flags.DEFINE_string("val_file", "small_val.csv",
                       "Annotations for validating video files")

tf.flags.DEFINE_string("output_dir", "/hdd4/lsmdc2017/records",
                       "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the videos.")

FLAGS = tf.flags.FLAGS

ClipMetadata = namedtuple("ClipMetadata",
                          ["clip_id", "filename", "sentences"])

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def feature_extract(image):
    
    return None

def _decode_video_to_images(video, sec=2, fps=24, is_randomly_spaced=False, feature_extractor=None):
    try:
        cap = cv2.VideoCapture(video)

    except:
        pass

    image_list = []

    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        image_list.append(frame)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_list = np.array(image_list)
    n_frames = sec * fps

    if frame_count > n_frames:
        if is_randomly_spaced:
            frame_indices = sorted(random.sample(range(0, frame_count), n_frames))
        else:
            frame_indices = np.linspace(0, frame_count, num=n_frames, endpoint=False).astype(int)
        image_list = image_list[frame_indices]

    if feature_extractor is None:
        return image_list
    else:
        print image_list
        return feature_extract(image_list)

def _to_sequence_example(clip, decoder, vocab):
    """Builds a SequenceExample proto for an clip-sentence pair.
  
    Args:
      clip: An ClipMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
  
    Returns:
      A SequenceExample proto.
    """

    jpg_file_dir = 'jpgs/' + clip.clip_id[:clip.clip_id.rfind('_')]
    encoded_image_list = []
    file_path = os.path.join(FLAGS.data_dir, jpg_file_dir)
    for file in os.listdir(file_path):
        with tf.gfile.FastGFile(os.path.join(file_path, file), "r") as f:
            encoded_image_list.append(f.read())

    try:
        for image in encoded_image_list:
            decoder.decode_jpeg(image)

    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid AVI data: %s" % clip.filename)
        return

    context = tf.train.Features(feature={
        "clip/clip_id": _bytes_feature(clip.clip_id)
        #"clip/data": _bytes_feature(encoded_clip),
    })

    assert len(clip.sentences) == 1
    sentence = clip.sentences[0]
    sentence_ids = [vocab.word_to_id(word) for word in sentence]
    feature_lists = tf.train.FeatureLists(feature_list={
        "clip/data": _bytes_feature_list(encoded_image_list),
        "clip/sentence": _bytes_feature_list(sentence),
        "clip/sentence_ids": _int64_feature_list(sentence_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_clip_files(thread_index, ranges, name, clips, decoder, vocab,
                        num_shards):
    """Processes and saves a subset of clips as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      clips: List of ClipMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_clips_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        clips_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in clips_in_shard:
            clip = clips[i]

            sequence_example = _to_sequence_example(clip, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_clips_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d clip-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d clip-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, clips, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each caption.
    clips = [ClipMetadata(clip.clip_id, clip.filename, [sentence])
             for clip in clips for sentence in clip.sentences]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(clips)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(clips), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, clips, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_clip_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(clips), name))


def _create_vocab(sentences):
    print("Creating vocabulary.")
    counter = Counter()
    for s in sentences:
        counter.update(s)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(os.path.join(FLAGS.data_dir, FLAGS.output_dir, FLAGS.word_counts_output_file), "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _process_sentence(sentence):
    tokenized_sentence = [FLAGS.start_word]
    tokenized_sentence.extend(nltk.tokenize.word_tokenize(sentence.lower()))
    tokenized_sentence.append(FLAGS.end_word)
    return tokenized_sentence


def _load_and_process_metadata(csv_file):
    with tf.gfile.FastGFile(csv_file, "r") as f:
        df = pd.read_csv(csv_file, header=None, sep='\t',
                         names=["clip_id", "start_aligned", "end_aligned", "start_extracted", "end_extracted",
                                "sentence"])

    id_to_filename = [(x, os.path.join(FLAGS.data_dir, x[:x.rfind('_')], x + ".avi")) for x in df["clip_id"]]
    id_to_filename = list(set(id_to_filename))

    id_to_sentence = {}
    for row in zip(df["clip_id"], df["sentence"]):
        id_to_sentence.setdefault(row[0], [])
        id_to_sentence[row[0]].append(row[1])

    assert len(id_to_filename) == len(id_to_sentence)
    assert set([x[0] for x in id_to_filename]) == set(id_to_sentence.keys())
    print("Loaded sentence metadata for %d clips from %s" %
          (len(id_to_filename), csv_file))

    print("Processing sentences.")
    clip_metadata = []
    num_sentences = 0
    for clip_id, filename in id_to_filename:
        sentences = [_process_sentence(s) for s in id_to_sentence[clip_id]]
        clip_metadata.append(ClipMetadata(clip_id, filename, sentences))
        num_sentences += len(sentences)
    print("Finished processing %d sentences for %d clips in %s" %
          (num_sentences, len(id_to_filename), csv_file))

    return clip_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    output_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir)
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    train_file = os.path.join(FLAGS.data_dir, FLAGS.train_file)
    val_file = os.path.join(FLAGS.data_dir, FLAGS.val_file)

    lsmdc_train_dataset = _load_and_process_metadata(train_file)
    lsmdc_val_dataset = _load_and_process_metadata(val_file)

    train_cutoff = int(0.85 * len(lsmdc_val_dataset))
    val_cutoff = int(0.90 * len(lsmdc_val_dataset))

    train_dataset = lsmdc_train_dataset + lsmdc_val_dataset[0:train_cutoff]
    val_dataset = lsmdc_val_dataset[train_cutoff:val_cutoff]
    test_dataset = lsmdc_val_dataset[val_cutoff:]

    train_sentences = [s for clip in train_dataset for s in clip.sentences]
    vocab = _create_vocab(train_sentences)

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
