# AIでりんごを描いてみた

## こんな感じ

### 生成結果(64x64 png)

![apple](docs/images/apple_500.png)  ![banana](docs/images/banana_500.png) 



### 途中結果

![apple](docs/images/apple_000.png) ite0

![apple](docs/images/apple_020.png) ite20

![apple](docs/images/apple_100.png) ite100

![apple](docs/images/apple_200.png) ite200

![apple](docs/images/apple_300.png) ite300

![apple](docs/images/apple_400.png) ite400

![apple](docs/images/apple_500.png) ite500



## 背景

最近GAN（Generative Adversarial Network）で、いろんな画像を自動生成したり、ポスターも生成するなどの記事はたくさん目に入る。

どこまで出来るかを試してみたかった。


## 実現の仕方

![gan](docs/images/gan.png)

こちら数字画像生成のgithubソースを参考し、改造した。 [2]

![mnist_digit](docs/images/digits.png)


## Usage


```
$ python3 origin-dcgan.py
```


## 参考

[1] <https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/>

[2] <https://github.com/osh/KerasGAN>

[3] <https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0>