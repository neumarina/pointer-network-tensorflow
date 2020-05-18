import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import TSPDataLoader

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.num_log_samples = config.num_log_samples
    self.checkpoint_secs = config.checkpoint_secs
    self.batch_size = config.batch_size

    if config.task.lower().startswith('line'):
      self.data_loader = TSPDataLoader(config, rng=self.rng)
    else:
      raise Exception("[!] Unknown task: {}".format(config.task))

    self.model = Model(
        config,
        inputs=self.data_loader.x,
        labels=self.data_loader.y,
        ds=self.data_loader.ds,
        o=self.data_loader.o,
        d=self.data_loader.d,
        save_keys=self.data_loader.save_keys,
        enc_seq_length=self.data_loader.seq_length,
        dec_seq_length=self.data_loader.seq_length,
        mask=self.data_loader.mask)

    self.build_session()
    show_all_variables()

  def build_session(self):
    self.saver = tf.train.Saver()

    # ckpt = tf.train.get_checkpoint_state(self.model_dir)
    #
    # # Define an init function that loads the pretrained checkpoint.
    # def load_pretrain(sess):
    #     self.saver.restore(sess, ckpt.model_checkpoint_path)
    # ,
    #                              init_fn=load_pretrain

    self.summary_writer = tf.summary.FileWriter(self.model_dir)

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    tf.logging.info("Training starts...")
    self.data_loader.run_input_queue(self.sess)

    summary_writer = None
    for k in trange(self.max_step, desc="train"):
      fetch = {
          'optim': self.model.optim,
      }
      result = self.model.train(self.sess, fetch, summary_writer)

      if result['step'] % self.log_step == 0:
        self._test(self.summary_writer)

      summary_writer = self._get_summary_writer(result)

    self.data_loader.stop_input_queue(self.sess)

  def test(self):
    tf.logging.info("Testing starts...")
    self.data_loader.run_input_queue(self.sess)

    for idx in range(1):
      self._test(None)

    self.data_loader.stop_input_queue(self.sess)

  def _test(self, summary_writer):
    fetch = {
        'loss': self.model.total_inference_loss,
        'pred': self.model.dec_inference,
        'true': self.model.dec_targets,
        'input_x': self.model.enc_inputs,
        'input_ds': self.model.aux_ds,
        'input_o': self.model.aux_o,
        'input_d': self.model.aux_d,
        'input_save_keys': self. model.aux_save_keys,
    }
    result = self.model.test(self.sess, fetch, summary_writer)

    tf.logging.info("")
    tf.logging.info("test loss: {}".format(result['loss']))
    fo = open('data/output.txt', 'w')
    for idx in range(len(result['pred'])):
      pred, true, input_x, input_ds, input_o, input_d, input_save_keys = result['pred'][idx], result['true'][idx], result['input_x'][idx], result['input_ds'][idx], result['input_o'][idx], result['input_d'][idx], result['input_save_keys'][idx]
      fo.write(';'.join([input_ds, input_o, input_d, str(input_x), str(true), str(pred), np.array_equal(pred, true), input_save_keys]))
      if not np.array_equal(pred, true):
        tf.logging.info("test pred: {}".format(pred))
        tf.logging.info("test true: {} ({})".format(true, np.array_equal(pred, true)))

    accy_ = [np.array_equal(result['pred'][idx], result['true'][idx]) for idx in range(len(result['true']))]
    tf.logging.info("accuracy: {}; len: {}".format(np.sum(accy_, dtype=np.float32)/len(accy_), len(accy_)))

    if summary_writer:
      summary_writer.add_summary(result['summary'], result['step'])

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
