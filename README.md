# SLAdehazing
<b>Self-supervised Learning and Adaptation for Single Image Dehazing</b>> (IJCAI-ECAI 2022 long presentation)</p>
Existing deep image dehazing methods usually depend on supervised learning with a large number of hazy-clean image pairs which are expensive or dif-
ficult to collect. Moreover, dehazing performance of the learned model may deteriorate significantly when the training hazy-clean image pairs are insufficient and are different from real hazy images in applications. In this paper, we show that exploiting large scale training set and adapting to real hazy images are two critical issues in learning effective deep dehazing models. Under the depth guidance estimated by a well-trained depth estimation network, we leverage the conventional
atmospheric scattering model to generate massive hazy-clean image pairs for the self-supervised pretraining of dehazing network. Furthermore, self-supervised adaptation is presented to adapt pretrained network to real hazy images. Learning without forgetting strategy is also deployed in self-supervised adaptation by combining self-supervision and model adaptation via contrastive learning. Experiments show that our proposed method performs favorably against the state-of-the-art methods, and is quite efficient, i.e., handling a 4K image in 23 ms.

## Environment settings
<pre><code>pytorch 1.5</code></pre>
We recommend using pytorch>1.0 and pytorch<1.6 to avoid unnecessary trouble. Our method does not rely on the special design of the network structure, so the rest of the general dependencies are not limited.

## train
Please wait for a while. We will tidy up the training code. It won't consume a lot of time.

## only-test
If you just need to test, you can execute the following code (need to load the parameters we provide):
<pre><code>python test_meta.py</code></pre>


It provides 2 different stages of test code, you can choose to test any labeled data set or unlabeled data and real haze maps on the Internet, our method does not need any labeled data during training, however, competitive results were still produced on these datasets.

It should be noted that the qualitative results of the first-stage results are better, and the quantitative results of the second-stage results are better on the real data set.

## checkpoint-files
You can get our parameter file from the link below
https://drive.google.com/drive/folders/1xq1tg7wvNJeZTw8w4RnqEsJzQRcmHLCI?usp=sharing

## test-data
We will give a link to the data we used for training, please wait while we sort out the training code.<br/>
The data used for testing can be obtained through the following links:<br/>
<em>SOTS:</em> http://t.cn/RQ34zUi<br/>
<em>4KID:</em> https://github.com/zzr-idam/4KDehazing<br/>
<em>URHI:</em> http://t.cn/RHVjLXp<br/>


