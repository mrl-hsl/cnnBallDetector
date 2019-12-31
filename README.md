Ball detection is one of the most important tasks in the context of
soccer-playing robots. The ball is a small moving object which can be blurred
and occluded in many situations. Several neural network based methods with
different architectures are proposed to deal with the ball detection. However,
they are either neglecting to consider the computationally low resources of hu-
manoid robots or highly depend on manually-tuned heuristic methods to extract
the ball candidates. In [this paper](https://2019.robocup.org/downloads/program/TeimouriEtAl2019.pdf), we proposed a new ball detection method for
low-cost humanoid robots that can detect most soccer balls with a high accura-
cy rate of up to 97.17%. The proposed method is divided into two steps. First,
some coarse regions that may contain a full ball are extracted using an iterative
method employing an efficient integral image based feature. Then they are fed
to a light-weight convolutional neural network to finalize the bounding box of a
ball. We have evaluated the proposed approach using a comprehensive dataset
and the experimental results show the efficiency of our method.
In this reposirory we have provided the source code of our CNN network. Also, the data set used for this work can be downloaded with the link below:
https://drive.google.com/drive/folders/13N7rVy0Evk3UmFOBqSjqSxAPrDgjOAFp?usp=sharing
