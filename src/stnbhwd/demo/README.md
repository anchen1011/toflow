# stnbhwd demo

Download MNIST and untar in the demo folder, then run with qlua (for image.display):

```
wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
tar -xf mnist.t7.tgz
qlua -ide demo_mnist.lua
```

Images should appear after 5 epochs and show what the STN does on a test batch.
You can edit demo_mnist.lua set use_stn = false to compare accuracy.

You will need to work with the getParamsByDevice branch of the 'nn' package (required for nn.Optim).
