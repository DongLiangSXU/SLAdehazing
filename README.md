# SLAdehazing
Self-supervised Learning and Adaptation for Single Image Dehazing (IJCAI-ECAI 2022 long presentation)

## Environment settings
<pre><code>pytorch 1.5</code></pre>

## only-test
If you just need to test, you can execute the following code (need to load the parameters we provide):
<pre><code>python test_meta.py</code></pre>

It provides 2 different stages of test code, you can choose to test any labeled data set or unlabeled data and real haze maps on the Internet, our method does not need any labeled data during training, however, competitive results were still produced on these datasets.

It should be noted that the qualitative results of the first-stage results are better, and the quantitative results of the second-stage results are better on the real data set.

## checkpoint-files
You can get our parameter file from the link below
https://github.com/DongLiangSXU/SLAdehazing/edit/main/README.md


