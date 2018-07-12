import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D
from keras.layers.convolutional import Conv2DTranspose as ConvT
from keras.preprocessing.image import img_to_array, load_img, list_pictures
import tensorflow as tf
from keras.backend import tensorflow_backend
from pylab import rcParams

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

rcParams['figure.figsize'] = 10, 10


class GAN():
    def __init__(self):
        # shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 潜在変数の次元数
        self.z_dim = 100

        d_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        g_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        # discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=d_optimizer, metrics=["accuracy"])

        # generator(コンパイルしない)
        self.generator = self.build_generator()

        # model結合
        self.combined_model = self.build_combined()
        self.combined_model.compile(
            loss="binary_crossentropy", optimizer=g_optimizer)

    def build_discriminator(self):
        activation = LeakyReLU(alpha=0.2)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2,
                         input_shape=self.img_shape, padding="same"))
        model.add(activation)
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(activation)
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(activation)
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(activation)
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()

        return model

    def build_generator(self):
        noise_shape = (self.z_dim, )
        activation = LeakyReLU(alpha=0.2)

        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation=activation,
                        input_shape=noise_shape))

        model.add(Reshape((16, 16, 128)))

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(activation)

        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(activation)

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        return model

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        model.summary()

        return model

    def train(self, epochs, batch_size=128, save_interval=50, X_train=[]):

        # change -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5)/127.5

        # dim2 -> dim3
        # X_train = np.expand_dims(X_train, axis = 3)

        half_batch = int(batch_size / 2)
        num_batches = int(X_train.shape[0] / half_batch)
        print("Number of Batches : ", num_batches)

        for epoch in range(epochs):
            for iteration in range(num_batches):
                # discriminator

                if iteration % 1 == 0:
                    # バッチサイズの半分をgeneratorから作成
                    noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                    gen_imgs = self.generator.predict(noise)

                    # バッチサイズの半分を教師データからピックアップ
                    idx = np.random.randint(0, X_train.shape[0], half_batch)
                    imgs = X_train[idx]

                    # training
                    # 本物と偽物は別々に
                    d_loss_real = self.discriminator.train_on_batch(
                        imgs, np.ones((half_batch, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(
                        gen_imgs, np.zeros((half_batch, 1)))

                    # それぞれの損失関数を平均
                    d_loss = np.add(d_loss_real, d_loss_fake) / 2

                # generator

                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                # 正解に近づけるようにする(本物ラベル、1)
                valid_y = np.array([1] * batch_size)

                # training
                g_loss = self.combined_model.train_on_batch(noise, valid_y)

                # 進捗の表示
                print("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))

            '''
            if not os.path.exists("./images"):
                    os.mkdir("./images")
            '''

            self.save_imgs(epoch)
            if epoch % save_interval == 0:
                '''
                if not os.path.exists("./h5"):
                        os.mkdir("./h5")
                '''
                self.combined_model.save_weights(
                    './h5/dcgan_origin_%d.h5' % epoch)

    def save_imgs(self, epoch):
        # 生成画像を敷き詰めるときの行数、列数
        r, c = 4, 4

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # 生成画像を0-1に再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5

        upper_limit = np.vectorize(lambda x: 1 if x > 1 else x)
        under_limit = np.vectorize(lambda x: 0 if x < 0 else x)

        gen_imgs = upper_limit(gen_imgs)
        gen_imgs = under_limit(gen_imgs)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)

        plt.close()


def main():
    row = 64
    col = 64
    channel = 3
    X_train = []
    img_list = glob.glob("data/train/apple/*.jpg")
    img_list += glob.glob("data/train/apple/*.jpeg")
    img_list += glob.glob("data/train/banana/*.jpg")
    img_list += glob.glob("data/train/banana/*.jpeg")

    for img_path in img_list:
        img = img_to_array(load_img(img_path, target_size=(row, col)))
        X_train.append(img)
    # 255で割ったりするのはtrain関数の中でやるよ
    X_train = np.asarray(X_train)

    gan = GAN()
    gan.train(epochs=1000, batch_size=32, save_interval=5, X_train=X_train)


if __name__ == '__main__':
    main()
