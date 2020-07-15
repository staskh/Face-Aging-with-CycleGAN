from __future__ import division
import argparse
import os
import time
from glob import glob
from collections import namedtuple
import numpy as np
import tensorflow as tf

import module


class cyclegan():
    def __init__(self, sess, checkpoint_dir, test_dir, dataset_dir, which_direction):
        self.sess = sess
        self.batch_size = 1  #
        self.image_size = 256  #
        self.input_c_dim = 3  #
        self.output_c_dim = 3  #
        self.L1_lambda = 10.0
        self.fine_size = 256
        self.ngf = 64  # G conv
        self.ndf = 64  # F conv
        self.output_nc = 3
        self.max_size = 50
        self.beta1 = 0.5  # adam
        self.epoch = 200  # 200
        self.epoch_step = 100  # lr
        self.train_size = 1e8  #
        self.lr_init = 0.0002  # lr
        self.load_size = 286
        self.save_freq = 500
        self.continue_train = True
        self.loaded = False
        self.inited = False

        self.checkpoint_dir = checkpoint_dir  # checkpoint
        self.dataset_dir = dataset_dir  # dataset
        self.test_dir = test_dir

        # G D
        self.discriminator = module.discriminator
        self.generator = module.generator_resnet  # resnet generator

        # loss_fn
        self.original_GAN_loss = module.mae_criterion

        self.which_direction = which_direction

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                      gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((self.batch_size, self.fine_size,
                                      self.ngf, self.ndf, self.output_nc,))
        # self.phase == 'train'))

        self._build_model()
        self.writer = tf.summary.FileWriter(self.checkpoint_dir, session=sess, graph=sess.graph)
        self.saver = tf.train.Saver()
        self.pool = module.ImagePool(self.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]  # self.real_data real A
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]  # real B
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")

        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False,
                                          name="discriminatorB")  # 32 BY 32
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        self.g_loss = self.original_GAN_loss(self.DA_fake, tf.ones_like(self.DA_fake)) \
                      + self.original_GAN_loss(self.DB_fake, tf.ones_like(self.DB_fake)) \
                      + self.L1_lambda * module.abs_criterion(self.real_A, self.fake_A_) \
                      + self.L1_lambda * module.abs_criterion(self.real_B, self.fake_B_)

        """    loss   
        #########################################################
        self.A_and_GA_hyunbo = self.generator(self.real_A, self.options, True, name="generatorB2A")
        self.B_and_GB_hyunbo = self.generator(self.real_B, self.options, True, name="generatorA2B")
        self.g_loss_by_hyunbo = self.L1_lambda * abs_criterion(self.real_A, self.A_and_GA_hyunbo) \
                                    + self.L1_lambda * abs_criterion(self.real_B, self.B_and_GB_hyunbo)

        #########################################################"""
        """ ===============================  =========================="""
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')

        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.original_GAN_loss(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.original_GAN_loss(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

        self.da_loss_real = self.original_GAN_loss(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.original_GAN_loss(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

        self.d_loss = self.da_loss + self.db_loss

        """============================================================================================"""

        """ test """
        self.test_A = tf.placeholder(tf.float32,
                                     [None, None, None,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, None, None,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            tf.logging.info(var.name)

    def train(self):
        """Train cyclegan"""
        self.lr_var = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        """
        #########################################################
        self.hyunbo_optim = tf.train.AdamOptimizer(self.lr_var, beta1=self.beta1) \
            .minimize(self.g_loss_by_hyunbo, var_list=self.g_vars)
        #########################################################
        """

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        counter = 1
        start_time = time.time()

        if self.continue_train:
            success, num_of_train = self.load()
            if success:
                counter = int(num_of_train)

        for epoch in range(self.epoch):
            dataA = glob('{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('{}/*.*'.format(self.dataset_dir + '/trainB'))
            tf.logging.info("num of dataA : %s" % dataA.__len__())
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), self.train_size) // self.batch_size
            lr = self.lr_init if epoch < self.epoch_step else self.lr_init * (self.epoch - epoch) / (
                    self.epoch - self.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [module.load_train_data(batch_file, self.load_size, self.fine_size) for batch_file in
                                batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, fake_B, _, g_loss = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_loss],
                    feed_dict={self.real_data: batch_images, self.lr_var: lr})

                # self.writer.add_summary(summary_str, counter)
                # [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, d_loss = self.sess.run(
                    [self.d_optim, self.d_loss],
                    feed_dict={
                        self.real_data: batch_images,
                        self.fake_A_sample: fake_A,
                        self.fake_B_sample: fake_B,
                        self.lr_var: lr
                    })
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                if counter % 10 == 0:
                    tf.logging.info(
                        f"Epoch: [{epoch:2d}] [{idx:4d}/{batch_idxs:4d}] "
                        f"time: {time.time() - start_time:4.4f}, d_loss={d_loss:.3f}, g_loss={g_loss:.3f}"
                    )
                    # tf.summary.scalar(name='g_loss', tensor=g_loss, step=counter)
                    # tf.summary.scalar(name='d_loss', tensor=d_loss, step=counter)

                if np.mod(counter, self.save_freq) == 2:
                    self.save(counter)
                    tf.logging.info("ckpt saved")

    def save(self, step):

        step_str = str(step)

        self.saver.save(self.sess,
                        self.checkpoint_dir + "/" + step_str,
                        global_step=step)

        """latest = tf.train.latest_checkpoint(checkpoint_dir)
        file_name = latest.split("/")
        file_name = file_name[-1]

        list = ['.data-00000-of-00001', '.index', '.meta']
        for x in list:

            drive_service = build('drive', 'v3')
            file_metadata = {
                'name': file_name + x,  
                'mimeType': None
            }

            media = MediaFileUpload('' + latest + x,
                                    mimetype=None,
                                    resumable=True)
            created = drive_service.files().create(body=file_metadata,
                                                   media_body=media,
                                                   fields='id').execute()
            tf.logging.info('File ID: {}'.format(created.get('id')))
            """

    def load(self):
        tf.logging.info(" [*] Reading checkpoint...")

        # model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            index = ckpt_name.find("-")
            num_of_train = ckpt_name[index + 1:]
            tf.logging.info(" checkpoint step : %s" % num_of_train)
            self.loaded = True
            return True, num_of_train
        else:
            tf.logging.info("check point load failed.")
            return False, 1

    def test(self):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.which_direction == 'AtoB':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testA'))
            tf.logging.info("dataset list loaded.")
        elif self.which_direction == 'BtoA':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testB'))
            tf.logging.info("dataset list loaded.")
        else:
            raise Exception('AtoB BtoA')

        if self.load():
            tf.logging.info(" check point loaded")
        else:
            tf.logging.info(" check point load failed")

        if self.which_direction == 'AtoB':
            out_var, in_var = (self.testB, self.test_A)
        else:
            out_var, in_var = (self.testA, self.test_B)

        for sample_file in sample_files:
            tf.logging.info('Processing image: %s' % sample_file)
            sample_image = [module.load_test_data(sample_file, self.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(
                self.test_dir,
                '{0}_{1}'.format(self.which_direction, os.path.basename(sample_file))
            )
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            tf.logging.info("start saving")
            module.save_images(fake_img, [1, 1], image_path)

    def deage(self, image_files, output_dir):
        if not self.inited:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.inited = True

        if not self.loaded:
            if self.load():
                tf.logging.info(" check point loaded.")

        out_var, in_var = (self.testA, self.test_B)

        for sample_file in image_files:
            tf.logging.info('Processing image: %s' % sample_file)
            sample_image = [module.load_test_data(sample_file, self.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(
                output_dir, os.path.basename(sample_file),
            )
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            tf.logging.info("Start saving")
            module.save_images(fake_img, [1, 1], image_path)
            tf.logging.info(f'Saved to {image_path}')

    def export(self, model_dir):
        if not self.inited:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.inited = True

        if not self.loaded:
            loaded, _ = self.load()
            if loaded:
                tf.logging.info(" check point loaded.")
            else:
                raise RuntimeError("Checkpoint cannot be loaded.")

        out_var, in_var = (self.testA, self.test_B)
        tensor_in = tf.compat.v1.saved_model.build_tensor_info(in_var)
        tensor_out = tf.compat.v1.saved_model.build_tensor_info(out_var)
        inputs = {in_var.name.split(':')[0].lower(): tensor_in}
        outputs = {out_var.name.split(':')[0].lower(): tensor_out}

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature
            })
        builder.save()
        tf.logging.info('Model saved to %s' % model_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'phase', choices=['train', 'test', 'eval', 'export'], default='eval'
    )
    parser.add_argument('--image')
    parser.add_argument('--output-dir')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/face_256')
    parser.add_argument('--dataset-dir', default='./datasets/face')
    parser.add_argument('--export-path')

    return parser.parse_args()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir  #
    test_dir = './test'  #
    dataset_dir = args.dataset_dir  # trainA trainB testA testB
    phase = args.phase
    which_direction = "BtoA"  # or BtoA.

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)  # gpu
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(
            sess, checkpoint_dir=checkpoint_dir, test_dir=test_dir,
            dataset_dir=dataset_dir, which_direction=which_direction
        )
        if phase == 'train':
            tf.logging.info("train")
            model.train()
        elif phase == "test":
            tf.logging.info("test")
            model.test()
        elif phase == 'export':
            if not args.export_path:
                raise RuntimeError('Provide --export-path to select model directory')
            model.export(args.export_path)
        else:
            if not args.image or not args.output_dir:
                raise RuntimeError("Provide --image and output dir for mode eval")
            model.deage([args.image], args.output_dir)


if __name__ == '__main__':
    main()
