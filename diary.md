# Diary

## CNN from scratch

11/03/2024: Succesfully trained small cnn on 4 cifar10 samples by having a very low learning rate and a lot (~1k) epochs.  
12/03/2024: better conv layer (looked at fft but ended up simply switching from einsum to tensordot)  
13/03/2024: 
  - max pool
  - Almost successfully trained on 7 samples cifar10.  
    Reached accuracy of 100% and then it seems like the gradients explode  
- Added layers weights means as metrics in traing stats df.  
  They seem to confirm that the gradients AND the weights are increasing and then exploding.  

17/03/2025:  
- Watched these videos about softmax:  
  - https://www.youtube.com/watch?v=ytbYRIN0N4g&ab_channel=ElliotWaite  
  - https://www.youtube.com/watch?v=p-6wUOXaVqs&ab_channel=MutualInformation  
- Learned that the aim of Batch norm layer is not to mean center and "unit variate" the inputs.  
  It's also an affine transformation after that.
  Its implementation is a bit more complicated than I thought it would be...
  Might switch to implementing the adam optimzer instead...
- Reread the NNFS chapters on optimizers
- Started implementing SGD and SGD_with_decay and refactoring layers to work with it.

18/03/2025:
- Tested refacto on mnist notebook, works fine :thumbsup:.
- Tested SGD_with_decay on 10 samples of cifar10, it converges int [400, 700] epochs and then diverges.
  I can't get it to stay at a minimum loss/accuracy...
  I will try to implement Addam to see if I can get it to converge and stay at a satisfactory loss minima/accuracy maximum.
  Actually I am first going to try to fit an nn with:
    - [Flatten, Conv, Softmax] This is to make sure that the issues I am encountering in the cnn's training are not a direct cause to some bad Conv implementation
      It's not working, it just stagnates or has big unexpected/e=unexplicable jumps in loss but never converges
    - [Flatten, Linear, Sigmoid] To see if the issues encountered in the training of the single conv layer cnn are caused by the Conv layer since a conv layer of the same shape as its input is (if I am not mitaken) the same as a Linear layer.
      And........... it's also not working -_-, it exhibits the same weird training patterns.
      I am assuming this is due to something else then, maybe the loss?
      By looking into it, it turns out that the gradients are so absurdly small that the substraction (that I have perfomed in a notebook just to be sure) gives the same param.
      *I also noted that the single Linear layer nn is ~40x faster than the single Conv layer nn so I will definetly look into (yet another) better Conv implementation.*

19/03/2025:
- Ok so it turns out that the learning rate was just too high, I set it to 0.03 and it works just fine with the [Flatten, Linear, Softmax] nn.
  However, it does not converge with the conv layer which, again is weird since it should be exactly the same thing...
  The [Flatten, Conv, Softmax] nn required a 0.0005 starting lr and a 0.005 lr decay where as the [Flatten, Linear, Softmax] nn didn't even need lr decay...
  I believe this is worth investing, let's see if this is caused by my implementation or an actual/real property of Conv layers.
  Ok, I checked with uncleGPT and it suggest (among other things thatI don't believe are worth looking into) that it might be due to the way gradients are computed.
  AND infact the gradients wrt kernels are computed as the full_convolve of the input and output I assuming that this is the reason for that diff and will move on to not get caught in, yet another, rabbit hole.
- Still, I can't get the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn to converge.
  I tried to tweak the starting lr and lr decay for 1-2 hours and it never got above 50% accuracy within ~400 / 1000 epochs.
  Currently it takes about 35 seconds to train the nn for 400 epcohs, lets try to improve this.
- I implemented a function that is ~30% faster than the tensordot current implementation with the 6 samples in the notebook.

20/03/2025:
- Testing the new cross corr implementation on 300 and 3000 samples, it is actually slower in both cases:
  at 300 samples: `@` and `tensordot` take the same amount of time but computing and flattening the views takes more time (obviously) than just computing the views.
  at 3000 samples: `@` takes more time than `tensordot`.
  So I will stay with the tensordot implementation.
- I will now get back to fitting the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn on htethe 10 samples cifar10 subset.
  Before starting to implement Adam I tried the following:
    - training a new nn 4 times, 
    - checking the iteration where the convergence point happens.
    - averaging the convergence iteration point
    - getting the learning cumsum up to this point
    - declaring a desired learning rate at this point as the learning rate at this convergence point dvided by an arbitrary denominator
    - computing the starting lr and lr decay based on the desired lr cumsum and desireed lr at the convergence point.
    I tried it multiple times but it never worked...
- I looked at [this notebook](https://www.kaggle.com/code/valentynsichkar/convolutional-neural-network-from-scratch-cifar10#Creating-Convolutional-Neural-Network-Model).
  Interesingly enough it uses only one 32 filters conv layer and two FC layers 
- Ok, I will start to implement Adam once and for all...
  Implemented SGD_with_momentum, it works, the mnist MLP converges faster with it... but it doesn't help me fit the nn to the cifar10 subset .
  I do not a different loss curve in the cifar10 subset nn training after the convergence point but nothing seems to change before...
  I will now start implementing AdaGrad.
  I implemented SGD_with_momentum and RMSprop.

21/03/2025:
- Implemented Adam optimizer.
  It improved the mnist score training accuracy from 0.93to 0.98!
  Damn it fitted the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn on the 10 samples cifar10 subset first try in 37 epochs wtf!!
- Watched a big portion of this [video](https://www.youtube.com/watch?v=l3O2J3LMxqI&t=3248s&ab_channel=MachineLearningStreetTalk)

22/03/2025:
- Started watching this [video](https://www.youtube.com/watch?v=7Q2JhZxNPow&ab_channel=SimonsInstitute)
- Getting ~60% on 100 samples cifar10 subset with the same nn and Adam
- Reached  80% by tweaking the Adam hyperparameters (mostly starting_lr and lr_decay).
    starting_lr=0.01,
    lr_decay=0.0001,
    momentum_weight=0.99,
    ada_grad_weight=0.99,
    epochs=200

23/03/2025:
- Watched almost all of this [video](https://www.youtube.com/watch?v=78vq6kgsTa8&t=663s&ab_channel=InstituteforPure%26AppliedMathematics%28IPAM%29).

24/03/205:
- Turns out that the nn doesn't always fits to ~80% on the 100 samples subset, most of the times it doesn't...
- Looking for ressources that explain **why** a network is not fitting.
  Find this [PDF](https://misraturp.gumroad.com/l/fdl), looks interesting.
- Going to try gradient clipping to see if it improves the training.
- Looked at plotly FigureWidgets to stream the training data to the figure. I will defenetly be using that instead of rich.progress.track + base plotly Figure/
- Reached 97% with the same hyper params as the 22/03/2025, interestingly enough, the abs gradient mean (almost) never went above 2. 
- Also notted that the undefitting nns seem to have  damped sin wave like loss curve where as the fitting ones seem to have a 1 - sigmoid like loss curve.
  This, hopefully, means that there is only one problem to solve, hoperfully...
- Started to refacto optimizers module to be smaller, more modular and use FigureWidget instead of rich.progress.track and then px.scatter
- FigureWdiget works fine BUT, it takes 2x more time to train an nn with it AND it doesn't render when you reopen the notebook so I might switch back to track + Figure.
  Though it is usefull to see the metrics live so idk I will leave it as is for now.

25/03/2025:
- Find out about dataclasses.dataclass and dataclasses.field, using them for the new Adam class.
- Read the 6.6 chapter of UnderstandingDeepLearning.
  Reading it, I learned that the point of having batches of the dataset at each step is to have a different gradients for the same model and avoid getting stuck in local minimas.
  > For nonlinear functions, the loss function may have both local minima (where gradi-  
  > ent descent gets trapped) and saddle points (where gradient descent may appear to have  
  > converged but has not). Stochastic gradient descent helps mitigate these problems.1 At  
  > each iteration, we use a different random subset of the data (a batch) to compute the  
  > gradient. This adds noise to the process and helps prevent the algorithm from getting  
  > trapped in a sub-optimal region of parameter space.  
  I tried to reduce the batch_size to 10 but it didn't seem to change anyhting...
  Maybe I should try to permutate the samples at each epoch to further randomize and the batch gradients(and maybe decrease momentum?)
  I also notted that the underfitting CNNs losses plateau at the same value(~3.27) which, I assume, is related to the distribution of labels in the 100 (first) samples.
  I will try to sample an equal amount of labels and see if I can more relaiably get a fitted model.

26/03/2025:
  - Made even x_train/y_train of 100 samples cifar10 subset doesn't seem to improve anything...
  - Implemented `nn_params_stats` and `activations_stats` metric functions to get their mean, std, l1 and l2.
  - I think the best way to understand why some models are underfitting and some others fitting, would be to get ~5 training stats (including activations, gradients AND params).
    And then try to find some meaningfull property that explains why some are fitting and some not. 
  - I will first implement a way to read/write a network to a json

27/03/2025:
  - Actually I first wrote a while loop that breaks once a fitting model has been found.
    The first time I ran it it got a 50% accuracy model but I messed up a line and the best model didn't get saved...
  - I re ran the loop a few times, trained abut 100 or so models, and obviou-fuckingly now no model is fitting (-_-).
  - Maybe I should do the study with models trained on the 10 smaples cifra10 subset instead of the 100 samples cifra10 subset as I get way more fitted models.
    Or speed up the training, if I could find a 10x improvement it would be enough to reliably find a fitting model in ~10mins... I think.
  - I looked at the AlexNet architecutre and it reminded that the number of kernels should increase as the are further in the model.
    My current architecture is:
    ```python
    [
        Convolutional((10, 5, 5, 3)),
        Relu(),
        MaxPool((2, 2)),
        Convolutional((10, 3, 3, 10)),
        Relu(),
        MaxPool((2, 2)),
        Flatten(),
        Linear(360, 64),
        Relu(),
        Linear(64, y.shape[1]),
        Softmax(),
    ]
    ```
    There are as many kernels as there are classes in the second conv layer wich might be problematic actually...
    I changed it to:
    ```python
    [
        Convolutional((20, 5, 5, 3)),
        Relu(),
        MaxPool((2, 2)),
        Convolutional((32, 3, 3, 20)),
        Relu(),
        MaxPool((2, 2)),
        Flatten(),
        Linear(1152, 64),
        Relu(),
        Linear(64, y.shape[1]),
        Softmax(),
    ]
    ```
    Let's see if that fixes it, it certainly takes forwever to train a model for 100 epochs...  
    6 minutes in and only 3 models trained, best accuracy is 12% -_-.  
    22 minutes in and the best accuracy is 14%...  
    While it is annoying that I can't find a fitting model anymore, the while loop is at least an effective way to test an idea.  
    After an hour and ~18 more trained models no improvements so I will switch back to the previous architecture.  
  - Let's try the btach norm layer.   
    Though, looking at the AlexNet architecture again,    
    I made chatGPT implement it, I was suprised to see that the mvg average/std, gamma and beta ar of shape [1, 1, 1, channels] and not [1, width, height, channels].   
    ChatGPT says that it's to preserve the signal that lies inbetween the values(he didn't phrase it that nicely, I did).   
    I guess that makes sense to me but following that logic it would also mean that there is a singal that lies in between the channels right?    
    Anyways, the results are unanimous, BatchNorm is amazing. Almost all the models fit.    
    Damn, I literally almost implemented it 10 days ago...   
  - I'm too tired to keep coding (or too lazzy idk...) so I watched this ["why batchnorm works" deeplearning.ai video](https://www.youtube.com/watch?v=nUUqwaxLnWs&ab_channel=DeepLearningAI).  
    Reminded me that I should look into deeplearning.ai.  
    It was great, made me understand why it has a small regularizatin but it didn't explain its effect on gradients.

28/03/2025:
  - I started implementing BatchNorm by myself because I didn't really like the xhatGPT implementation, even though it works jsut fine.  
    I wanted to implement the normalization as `(inputs - mean) / (std - epsilon)` instead of `(inputs - mean) / sqrt(variance + epsilon)`.  
    The latter was in the original paper of the batchnorm layer so I thought that there was probably a reason for the presence of the sqrt.
    I asked chatGPT if these were equivalent and it turns out that it isn't.  
    The latter is better because its output for small variance inputs is properly mean centred (the mean value is the same but numerical instability causes very small numbers to not get mean centerd output).  
    And the outputs variance is also closer to one for the second method.  
    I'm not really sure it would impact that much the training but let's not take the risk just to remove one sqrt call, shall we?  
  - Updated the repo with all the modifications and rewrote BatchNorm to my liking, I will try to perform batchnorm over other axis combinations than "just" (0, 1, 2) and apply it on the fc layers (and input layer?).  

29/03/2025:
  - The network works fits properly on 1000 samples cifar10 subset, though it takes forever to train (~10 mins).
  - Trying on 5k samples, it fits up to ~50% accuracy and then the accuracy (and loss) plateaus.
  - I might need to switch implementation because to speed things up, it's a shame there isn't a simple numpy on GPU alternative, I looked into numba and it seems to have a lot of caveats...
  - Looking at the cifar10 from scratch kaggle notebook I (re)saw that the model had only one (bigger, 32x7x7x3 wher I have 10x5x5x3) conv layer.  
    And a hidden FC layer of output size 300.
    I.e a shorter wider network.
    I tried to mimicking this by removing the second conv layer and set the ouput size of the hidden FC layer to 128.
    It works!
    I notted that the output size of the first FC hidden layer (64) often turned out to be a bottleneck.

31/03/2025:
  - Maybe I should switch from `tensordot` to `scipy.signal.convolve` which under the hood can use `fft`.
  - The kaggle cifar10 with numpy kaggle notebook uses a beta_1 of 0.8 (which corresponds to the `momentum_weight` in my Adam optimazation).
    I tried the same value for my training and it did speed things up: the model plateaus to 50% earlier.  
    It's not a real improvement but at least it should help me find a solution faster as it will take me less time to check if a solution works.  
  - I added back the second conv layer and it seems to work a lot better, reaching 72% accuracy at epoch 50.
  - I increased the number of filters in the second conv layer from 10 to 32 but it didn't seem to really increase the perfs. 
  - Reached 85% accuracy after 100 epochs... and 30ins.
  - I will try to decrease tthe number of filters in the second conv layer to 20 and increase the size of the filters of the first from 5x5 to 7x7.
  - Even with this fairly high accuracy on this training data subset, I only get 33% accuracy on the test data.
    That is the same score as the nn trained on the 1k samples subset.
    That being said the x_test is 10k samples so it's not that suprising.

01/04/2025:
  - Looked into a implementation of [cnn on cifar10 with cuda](https://www.kaggle.com/code/alincijov/cifar-10-numba-cpu-cnn-from-scratch) but it didn't seem any faster.
    > Note: I had to fix this line `for i in range(len(df1)):` into `for i in range(len(df1[0])):` (and turn the internet of the notebook on but this is not a fix).  

    So maybe I shouldn't work with numba?
    I found another [cnn with cuda](https://github.com/WHDY/mnist_cnn_numba_cuda/tree/master) let's look into that.  
    The (numpy) kaggle notebook used only 10 epochs so I should be able to get away with numpy and my desktop.   
    Maybe I should look into the weight initialization and try to replicate all the settings to see there is an improvement I can find and a lesson to learn from it.  
    Actually it has the same weight initialization as me...
    Let's try to set the hidden fc output size to 300.

02/04/2025:
- Actually, now that I think about it the full dataset is 50k i.e 10x the size of my 5k subset so it's not shocking that the numpy kaggle notebook needs 10x less epochs.
  - I'm going to look into CuPy, it also seems interesting
  - I tried a bunch of chatGPT written numba versions of my valid correlate but none of them were faster(sad)...
  - I tested the numpy cnn kaggle notebook and it is super slow.  
  - So back on the cupy option I would need to setup a remote session... let's try on kaggle.  
    Looking at this [cupy CNN repo](https://github.com/AlaaAnani/CNN-from-Scratch-Cupy), I stumble upon the lealkyReLu, I wander if this could speed up my training.  
    Letme ask daddyGPT and uncle Google.  
    Seems like no.... but I might try it anyway later.  
  - Saw that cupy does not provide `sliding_window_view`, but it does provide `as_strided` which I believe can be used for the same goal.   
    In fact, looking at the source of `sliding_window_view` we can see that it is built on top of `as_strided`:thumbsup:.  
  - As expected, setting up the kaggle notebook is an absolute pain in the ass...  
    Gonna try lightning AI.  
    Tried it, setup was pretty straight forward which is really cool.    
    However, when I tried this chatGPT written code snippet:  
    ```python
    import numpy as np
    import cupy as cp
    import time

    # Generate random inputs
    views_np = np.random.rand(10000, 26, 26, 3, 7, 7)
    k_np = np.random.rand(32, 7, 7, 3)

    # NumPy tensordot benchmark
    start = time.time()
    np_result = np.tensordot(views_np, k_np, axes=([3, 4, 5], [3, 1, 2]))
    np_time = time.time() - start
    print(f"NumPy tensordot Time: {np_time:.4f}s")

    # Transfer to GPU (CuPy)
    views_cp = cp.asarray(views_np)
    k_cp = cp.asarray(k_np)

    # CuPy tensordot benchmark
    start = time.time()
    cp_result = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
    cp_time = time.time() - start
    print(f"CuPy tensordot Time: {cp_time:.4f}s")

    # Transfer back to CPU for comparison
    cp_result_np = cp.asnumpy(cp_result)

    # Validate correctness
    assert np.allclose(np_result, cp_result_np, atol=1e-5)
    print("Results match!")
    ```
    The time for numpy and cupy were equal (1.3s).  
    That suprised me in two ways:
    1.  Why isn't cupy faster?!!!!!
    1.  1.3s seems pretty fast for numpy letme compare this to my computer.  
    Ran it on my computer and it's infact a lot slower: 5s.  
    So even if I don't use the lightning GPU I would still get a 5x speed increase without any overhead caused by switching to cupy.   
    That sounds really good.  
    Started to run the cifar10 notebook on the remote ligthning computer but it seems to run at the same speed with 14 epochs in 7 mins same as my computer...  
    At 14 mins, my computer actually outran the lightning AI computer: 32 epopchs vs 28 respectively.  
  - Turns out that cupy also has a warmup, let me test it a second time.
    Updated the code:
  ```python
  import numpy as np
  import cupy as cp
  import time

  # Define the input shapes
  views_shape = (10000, 26, 26, 3, 7, 7)
  k_shape = (32, 7, 7, 3)

  # Generate random input data
  views_np = np.random.rand(*views_shape)
  k_np = np.random.rand(*k_shape)

  # Convert to CuPy arrays
  views_cp = cp.array(views_np)
  k_cp = cp.array(k_np)

  # Warm-up CuPy
  _ = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
  cp.cuda.Stream.null.synchronize()

  # Measure NumPy time
  start = time.time()
  tensordot_np = np.tensordot(views_np, k_np, axes=([3, 4, 5], [3, 1, 2]))
  numpy_time = time.time() - start
  print(f"NumPy tensordot Time: {numpy_time:.4f}s")

  # Measure CuPy time with multiple iterations
  iterations = 10
  start = time.time()
  for _ in range(iterations):
      _ = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
  cp.cuda.Stream.null.synchronize()
  cp_time = (time.time() - start) / iterations
  print(f"Average CuPy tensordot Time: {cp_time:.4f}s")
  ```
  Ok, it's better:
  - numpy: 1.3s
  - cupy: 0.5s   
  I was expecting a bigger improvement tbh, like a 10x...  
  Let's see if uncleGPT can make it better.


03/04/2025:
  - Switching the batch size from 200 to 500 (tried to see if it would speed things up), made the model untrainable.  
    This modification was not pushed to the remote repo, maybe it did speed things up and that's why the remote lightning AI computer took more time to perform the epochs?  
    I'm running the training again with 200 batch size on my desktop let's see how that will change the speed.  
    No, it's, suprisingly enough, faster to train with smaller batch size: 33 epochs in 13 mins this time on my desktop.  
    Ok, I think I know why there was a speed diff, I also forgot the change to the architecture of the network.  
    Let's rerun the notebook on the remote computer to see if there is any diff.  
    At ~4 mins, it looks like it didin't change anything...  
    No diffs indeed...
  - Maybe I don't need to speed things up after all?
    Let's say it takes me ~2 days to get the best off of cupy and get a ~10x improvement it would be nice.  
    BUT, it might prevent someone who doesn't have a cuda GPU to run the notebook.  
    AND maybe I can just run the training on the 50ks which should take ~10x more time (5 hours) to train that give me ~two trained models per night.  
    That way I can focus on saving on the model.  
    Then I would move on to pytorch, I feel like I would be wasting time if I kept on building on top of numpy.
  - Let's try to scale up to 10k samples cifar10 subset, same architecture/hyperparameters.  
    First run did not succeed, I changed the batch size to 500 (to try to speed things up) and decreased the starting leargnig rate from 0.025 to 0.015.  
    It works!  
    80% accuracy at epoch 28 in ~22mins!  
    Weirdly enough, doubling the dataset size sped up the training...  
    Test accuracy raised to 40% test accuracy. 
  - Let's try the full dataset then (I might need to increase the swap)...
    I encountered this error: `MemoryError: Unable to allocate 37.0 GiB for an array with shape (50000, 26, 26, 3, 7, 7) and data type float64`.  
    Given the shape of the ndarray, I assume this is the ndarray of the views.  
    What's weird is that the sliding_window_view should only create a view of the input tensor.  
    Ok, looking at the traceback, I see that this is because the tensordot function uses a transpose on the views before performing the dot call.  
    This effectively requires copy (and therefore an allocation) of the views to be made.  
    Since this problem arises only for metric recording of the full dataset loss/accuracy I will simply split the forward call on the entire dataset in calls of 10k subsets of the dataset and then concat them.  
    That seems to have fixed it.
  - Three epochs and 4 mins in and we are already at 50% accuracy (on training set) I'm actually mind blown.  
    I wander if einsum would be faster... if it doesn't use transpose, it would not need to do all those allocations... to be continued.  
  - While the training continues I will get a beer...
  - OOOOOOOOOOOOOOOOOOO it fitted the full training dataset at 89% accuracy!!!!!!
  - Test accuracy is 50% but I think this is only because I left the training going on for too many epochs.
    I'll find out tomorrow

04/04/2025:
  - Interrupting the training at epoch 28, we get a train/test accuracy of 70 and 60 percent respectively.    
    This confirms that the model of yesterday did overfit.  
  - Trying to switch back to einsum to see if it speeds things up (looks like it doesn't).
    I tested them with `timeit` and `tensordot` is, in fact, faster.
  - Looked at a lot of videos on GPUs to see which one could fit me best, the 5070 or some GPU of the 30 series seem like the best options.  
  - I have also looked at cloud GPU options and runPod and Vast.ai seem pretty interesting.
  - Looked at the kernels and they resemble the cpu kaggle's kernels which is a good sign.

06/04/2025:
  - Retrained a model for 30 epochs with 15 kernels and interestingly enough, only 5 kernels are used.  
    By "not being used", I mean that the sum of the activation maps of a 1000 samples batch over the 0(sample), 1(width) and 2(height) axes are 0 after Relu.  
    I believe this is known as a dead neuron, although here it's more like a dead kernel.  
  - Note that I never(~5 trainings) experience this with 10 kernels models and always (with ~3 trainings) do with 15 kernels models.  
    So maybe this is due to a missimplementatin (in the batchNorm maybe)?
  - So let's try leakyRelu:manshrugging:.

07/04/2025:
  - LeakyRelu had a typo that made the gradient innacurate.
    After fixing it and switching Relu by LeakyRelu after the conv/BatchNorm layers, I trained the 15 kernels models for 22 epochs.  
    Got a training and test accuracy of 69 and 61 percent respectively.  
    This time all the filters are used.  
    I looked at the hidden Fc layer activations for 1000 imgs and the activations were also VERY sparse.  
    I will try replacing the second ReLu with LeakyReLu too.  

08/04/2025:
  - I ran a 35 epochs training on the LeakyRelu model but there was no improvement...

09/04/2025:
  - Tried l2 norm but it didn't seem to work...

10/04/2025:
  - Trying to use the LeakyRelu only on the hidden fc layer.
  - Saw in the kaggle notebook that the conv layer has a stride of 2 and a padding of 1, that something that might be worth looking into.  


15/04/2025:
  - Implemented a `sliding_window_view` on top of as_strided that works like the numpy function with the addition of the stride parameter.  
  - Implementing that function is just the tip of the iceberg tho.
    To update the entire network, I would also need to learn how the backward pass would change with strides and padding in the forward pass.  
    That's something I should look into but it would take some time I would rather look into regularization techniques like dropouts and take another look at l2 norm/weight decay.  
  - Watched [this video](https://www.youtube.com/watch?v=q7seckj1hwM&ab_channel=ArtemKirsanov) I didn't understand it entirely but it was imtersting and helped a little bit to understand l1 and l2 norm.

16/04/2025:
  - Made chatGPT implement a droput layer (booooh chatGPT I know, I know...)
  - Added dropout layer after maxpooling layer resulting in the following architecture:
    ```python
    [
        Convolutional((10, 7, 7, 3)),
        BatchNorm(),
        Relu(),
        MaxPool((2, 2)),
        Dropout(0.25),
        Flatten(),
        Linear(1690, 300),
        LeakyRelu(),
        Linear(300, y.shape[1]),
        Softmax(),
    ]
    ```
  - Started a training run for the night.

17/04/2025:
  - The test accuracy went to 61% so.... not that big of an improvement(sad).

24/04/2025:
  - I think this is going to be the end of this project, it's a shame I didn't get a better test accuracy.  
    I'm stopping here because I thnik I am spending time on waiting for the result of my improvement attemps.  
    I think I should have used something like `cupy` and implement an auto gradient computing.   
    Maybe I would get back to it later but now I need to start using `pytorch` and to learn high level concepts rather than wait for the training to finish...  
    And learn an actually used library.


25/04/2025:
  - Actually...... I relooked into cupy and thought that maybe I had done something wrong in my test because the other online comparison I could fine had much (much) higher speed ups.
  I guessed that the small 3x speed could be due to the `tensordot` call.   
  So I ran another `cupy` code test:
  ```python
    import numpy as np
  import cupy as cp
  import time

  # Print available devices
  props = cp.cuda.runtime.getDeviceProperties(0)
  name = props['name'].decode('utf-8')  # bytes → str
  major = props['major']
  minor = props['minor']
  total_mem = props['totalGlobalMem'] / (1024**3)
  print(f"  Device {0}: {name}  (Compute {major}.{minor}, {total_mem:.1f} GB)")


  # Define the input shapes
  views_shape = (10000, 200)
  k_shape = (200, 32)

  # Generate random input data
  views_np = np.random.rand(*views_shape)
  k_np = np.random.rand(*k_shape)

  # Convert to CuPy arrays
  views_cp = cp.array(views_np)
  k_cp = cp.array(k_np)
  print(views_cp.device)
  # Warm-up CuPy
  _ = views_np @ k_np
  cp.cuda.Stream.null.synchronize()

  # Measure NumPy time
  start = time.time()
  tensordot_np = views_cp @ k_cp
  numpy_time = time.time() - start
  print(f"NumPy tensordot Time: {numpy_time:.4f}s")
  
  # Measure CuPy time with multiple iterations
  iterations = 10
  start = time.time()
  for _ in range(iterations):
      _ = views_cp @ k_cp
  cp.cuda.Stream.null.synchronize()
  cp_time = (time.time() - start) / iterations
  print(f"Average CuPy tensordot Time: {cp_time:.4f}s")
  ```

  And got the  following result:
  ```
  Device 0: Tesla T4  (Compute 7.5, 14.7 GB)
  <CUDA Device 0>
  NumPy tensordot Time: 0.0840s
  Average CuPy tensordot Time: 0.0011s
  ```

  Now that's more like it!!!
  that's a 76x speed up!!!!!!
  Let's say I get a 50x speed up over an epoch that would reduce my 35 epochs training time from ~3h to ~4 mins!!!.
  - Excitement is back, I'm glad I gave a last look to the project before moving on!
  - Now I need to find yet another cross corr implementation that gives such a speed up since the tensordot is clearly not an option.
  - Hold that thought, I am hvaing a ophthalmic migraine...
  - Ok I'back... and dizzy...
  - I tried the `cupyx.profiler.benchmark` function, it's really good.
    It's telling me that the `cupy.tensordot` function is mostly ran on the GPU.  
    I looked at the source code but it's a lot more obscure than the numpy implementation.   
  - I gave a look at the [cnn with cupy repos](https://github.com/AlaaAnani/CNN-from-Scratch-Cupy) that I had found a few weeks ago.
    In it, the convolution is implemented witht the im2col method.
  - I had to look at this [medium post](https://medium.com/@sundarramanp2000/different-implementations-of-the-ubiquitous-convolution-6a9269dbe77f).  
    It confirmed my guess that the im2col cross corr method is the same as the one I first implemented, I guess I really am a genius lol...(I'm not...)  
  -I am going to switch back to that method but this time with `as_strided`+`reshape` call instead of the intricate, lengthy index creation method.  

26/04/2025:
  - I have (re)implemented the new im2col correlation, works just fine.
    However it (again) holds a 3x speed up -_-
  - Also the benchmarking function is a bit odd I don't always get results that make sense
  - Found this very interesting [notebook on github](https://github.com/anilsathyan7/ConvAcc/blob/master/conv_acc.ipynb).
    The author gets a 30x speed up.  
    Tho his method is not vectorized for multi inputs, kernels, channels.

27/04/2025:
  - Improved speed up simply by reducing batch size.

28/04/2025:
  - Implemented proper function to benchmark the computations of a given inputs and kernels shape.  
    And convert the results into a dataframe. 
  - Using timeit + cp.asnumpy gives time measurments that make more sense.

29/04/2025:
  - Found a github issue that suggest using float32 instead of float64, after performing the switch a get a 6x speed up which is pretty cool.
  - Added the corss corr acc implementation of the [github notebook](https://github.com/anilsathyan7/ConvAcc/blob/).
    Interestingly enough, the corss corr acc implementation is faster than mine even tho the implementations are very similar.  
    It seems like there are some hidden overhead caused by the abstraction 

30/04/2025:
  - Ran a bigger benchmarking with more input/kernels shapes combinations.  
    Clearly, there is a weird cupy overhead because some times I see the GPU at 100% and some times at ~30%.  
    The speed ups go from 0.5x (so actually slower than numpy) to 40x.
    Note: I run the computation five times for each combination.


## Deep Learning Computer Vision training and interpretation

14/05/2025:  
-   Found out that batch size greatly affects GPU speed up over CPU.  
-   Currently I have a modified copy of the CIFAR10 pytorch notebook.  
    I modified it to understand how to train a image classifier on a GPU.  
-   Now that I have the bare minimum working notebook, I want to clean things up.  
    I am going to use the code I wrote in my [cnn_from_scratch](https://github.com/MauroAbidalCarrer/CNN_from_scratch) to put all the boiler plate code in seperate modules.  
    This way I can also get the live view of the charts of the metrics over epochs during training.  
    I considered using some higher level library like pyorch lightning but I figured I would first get my hands "dirty" then move on to those kind of libraries.  

17/05/2025:
-   Realized that there was an error in the `Trainer.metrics_of_dataset` function.  
    Turns out that it was actually expressing the accuracy as the mean of the sums of correct outputs per batch.  
-   Switched from  conv+relu+fc+relu to conv+relu+batchNorm+fc+relut+softmax type of model.  
    Switched from SGD (with momentum even if it's not written in the name) to Adam.
-   Came across this very [interesting notebook](https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min).  
    I still need to (better) understand why/how the res net work for classification tho.  
-   I read the (first, classification half of) VGG paper to understand why are conv blocks are used.  
    From what I understand, conv blocks are stacks of small (in the paper 3x3 stride 1 and pad 1 to maintain the same width/height of the input) conv layers.    
    They essentially "emulate" what a wider single conv layer would do.  
    Let's take the example of the paper of a block of 3 3x3, stride:1, pad:1 convs layer vs a 7x7 stide:2 both with input channel count = output channel count.  
    The conv block comes with these added benefeats:
    -   Less parameters the conv  block would have 3*(3\*\*2\*C\*\*2) (C squared because Cin = Cout) = 27C\*\*2 against 7\*\*2C\*\*2 = 49C\*\*2.  
        This is a 1.8x decrease in size.
    -   More non linearity since we have two more relu layers.  
    -   Better regularization/generalization as the 3*3 is a form of decomposition of the 7x7.  
        The way I understand this is that the representations that a block can learn cannot be as tied to the training data as the single 7x7 conv layer.  

    Note that the conv block and wide conv layer have the same receptieve 7x7 field.  
    I don't understand why the width/height and channel count respectively decrease and increase in between blocks instead of decrease by layer.  
    I asked chatGPT and it said that it's to preserve spatial information which sounds weird since the input will either zero or same padded...  
-   Started to read the Network in Network (NiN) article of the Dive into deep learning and found out about the lazy 2d conv in pytorch.

19/05/2025:
-   Read the dive into deep learning article of resnet.  
    Interstingly enough they say that the reason for the skip connection creation is not the vanishing/exploding gradient problem but rather non nested function groups.  
    Meaning that bigger networks can't necessarily do what smaller networks can do.  
    That sounds odd tbh.  
    One of the comments points that out and a response says that this simply what the original paper says.  
    I will try to read it tomorrow.  
-   Abondend the ida of doing the leaffliction project, instead I will "simply" train a model on a (hopefully) big data set and then try to reimplement some interpretation paper.  

20/05/2025:
-   Read the paper on the resnet architecture.  
    It was interesting.  
    It does in fact, state that the probleme it tries to solve is the "degradation problem" and not the "vanishing/exploding gradients".  
    The most interesting thing I learned is that not only does the resnet arcitecture allows for deeper models it also allows them to be a LOT thinner than VGG architecutre.  
    In the paper, they use the resnet-34 (34 layers) and VGG-19(you can guess what the 19 means...).  
    While the resnet-34 has more layers it takes 18% of the FLOPs that VGG-19 takes.  
    that is mostly because the conv layers need a lot less channels.  
    And also because there is only on fully connected layer at the end of resnet-35 against 3 of output size 4096 for VGG-19.  
-   Ok, now I can finally start reimplementing the CIFAR10 90% in 5min kaggle notebook.  


21/05/2025:
-   Read the web version of [DNNs always grok and here is why](https://imtiazhumayun.github.io/grokking/).  
    It was very interesting, hopefully I can reimplement the Local complexity measure.  
    Ideally I could use it as input for some sort of training policy.  
-   I finished the first reimplementation of the kaggle notebook but the model is super slow.  
    Then the remote machine crashed.  
    So hopefully the issue comes from the machine and not the code.  
    I ran the code on another machine and it still is super slow: one hour for two epochs...  
    Which is odd since the notebook is called "CIFAR10 90% in 5 mins".  
    Now I actually hope that there is something wrong with my code.  
    If not, it means that I will have to pay for a better, more expensive, GPU.  
    I fixed the kaggle notebook by replacing the code cell that downloaded CIFAR10 from fast.ai by a cell that downloads it using torchvision.datasets.CIFAR10.  
    Then I ran it on kaggle using a P100 GPU.  
    It trained the model in 2 mins(wtf?!!).  
    I downloaded the notebook on the schools computer and addded it to the repos.  
    I pull the notebook from the repos onto a vastai instance with a 4090.  
    I runs faster than my reimplementation: 2 epochs in 13 minutes.  
    But that's nothing compared to the 8 epochs in two minutes.  
    So either I switch my workflow from vastai to kaggle OR I search for a simple opti trick.  
    I looked for FFT 2D conv but I couldn't find an pytorch API reference for it.  
    Also the forums seem to suggest that the benefits of using FFT for convolution emerge when using much larger filters amd inputs.  
    I ran the same notebook on a A100 vastai instance and the 8 epochs training took one minutes.  
    Damn...
    I tested the notebook on a Tesla V100 and it ran in 1min16s but it costs 28¢/h instead of the ~1$/h for the A100.  
    So I'll defenetly be using that going forward.  
-   Tommorow I will try to understand why my trainer implementation is slower than the kaggle notebooks implementation.  
    And Then I will have to add in all the other features like learning rate scheduling.  

22/05/2025:
-	I updated the setup_linux_machine repo to increase productivity.  
	I added an aliases.zsh file that contains all the aliases I already had + `p` and `amp` git aliases.  
	I might also use a repo I found that manages the .ssh/config file automatically.  
	It looks like I will do anything to not work on the "main quest" of this repo lol.  
    I tried the vastai cli and the vastai-ssh-config package I found online but I coulnd't make them work so gave up on that.  
-   Now the "real" work of the day begins, I am going to try to find out why my code is slower than the original one.  
    Turns out the model was simply not on the GPU I just had forgot to add a .cuda() call to its declaration.  
    Nevertheless, my training still runs two times slower than the original one (2m15s instead 1min15s).  
    Even tho this is not a very big diff for me right now, it's worth investigating into it to learn from it now.  
    It also turned out to be a simple reason: 
    I would recompute the outputs of the model with `no_grad` on the training set where the kaggle notebook uses the outputs of the training loop.  
    According to chatGPT, this is the conventional way of doing it.  
    I set out to update the Trainer implementation... and it took me an embarassing amount of time(hours).  
    I ended up with a convoluted solution, which I actually threw away for a simpler uglier solution but at least it works with minimal modifs.  
    Now I can start adding the "fancy" features.  
    Namely, lr scheduling, weight decay and gradient clipping.  
    Done.

23/05/2025:
-   I read this [toturial/explanation web page](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) on how to find the right learning rate.  
    Very intereseting.
    KTs:
    - To find the learnging rate:
        - Start training with a small learning rate.
        - At each iteration, register the smoothed exponentially weighed averaged loss and muiltiply it by some hyper parameter for the next iteration.
        - The loss will decrease then increase.
        - Once it increases to four fold that of the minimum registed loss stop the "learning rate search"(that expression comes from me).
        - The web page states that the learnig rate with the lowest loss divided by 10 should be taken.  
          The reason for taking a learning rate smaller than the one with the lowest loss instead of taking the later is because of the exponentially weighed averaging: it makes the loss rise later.  
          I thought it was kinda weird to take the learning rate with the lowest loss since it is preceeded by other training steps that have already deacreased the loss in the previous iterations.  
          I asked why not train the model from start for each learning rate to chatGPT.  
          It said that while it would be accurate, it would be a lot more compute intensive and that searching for the learning rate in a single training run is good enough.  
          It also said that sometimes the learning rate with the highest loss difference (compared to the previous iteration) is chosen.  
          That makes a lot more sense.  
-   Then I read this [toturial/explanation web page](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy) of the same author on one cycle policy.  
    This is a learning rate scheduler.  
    In fact this is the page I wanted to read in the first place. However it was refering to the previous page which looked important.  
    I didn't learn THAT much mostly because there wasn't much content on the page.  
    KTs:
    - The scheduling is comprised of three parts:
        - A linear ascent of the learning rate from a low value (usually 10x lower that the value found by the lr finder) to the "normal" value, the one find by the learning rate finder.    
          This is know as the warm-up phase and this is what allows us to take a higher learning rate than if we didn't go through this warm up phase.   
          The ascent takes ~half ot the training.  
        - A slightly shorter long linear descent from the found learning rate back to the low value.
        - Yet another linear, very short descent to ~100x less than the starting value (so ~1000x less that the one found by the LR finder).  
    I don't really understand what the warm up is so I will have to into that another day.  
-   I read a reddit blog post that asked why the learning rate was used and why some papers say that it is usefull while others say that it is not.  
    The only response that seemed to make sense is that it prevents the Adam optimizer from accumulating early mostly random gradients in its momentum.  


25/05/2025:
-   I watched this [video] (https://www.youtube.com/watch?v=KOKvSQQJYy4&ab_channel=Tunadorable) on the warmup of learning rate.  
    KTs:
    - New terms:
        - loss sharpness: How much the learning rate changes for a given change in parameters.
        - trainging/loss catapults: The oscillations of the loss during training.
    - There are two reasons for the success of learning rate: 
        1.  The one discussed in the reddit post, it prevents Adam from accumulating noisy initial gradients in its momentum.  
        1.  The loss landscape is at first very sharp.  
            -   If we use a normal decaying learning rate policy/schedule we will essentially find the closest crevasse in the landscape.  
                This is not ideal since crevasses in sharp landscapes are usually overfit regions.
            -   If we first warmup the learning rate, the model will escape this sharp region and arrive at a flatter region.  
                This is great for two reasons:
                -   Flat regions are usually regions of generalization.
                -   They allow us to use higher learning rate.
        > Warning: I passed the "sharp to flat LR warm up theory" to chatGPT to confirm and it denied it once out of three times...
    - The warmup is not always necessary since the loss landscape is not always sharp it's mostly used on large models or for trainings with large batch sizes.  
    KT from chatGPT:  
    - The warm-up also helps the batch/layernorm layers to initialize their parameters similarly to the way warm up helps to initialize the momentum of Adam.  
-   With all of that out of the way, I can start looking into the interpretation papers/blog posts.  
-   I started reading the [Deep Networks Always Grok and Here is Why](https://arxiv.org/pdf/2402.15555) paper.  
    It's pretty verbose in it's introduction.  
    I just got started, but I learned a few new mathematical terms.

26/05/2025:
-   I finished reading the paper.  
    KTs:
    -   Terms:
        -   Delayed generalization: aka Grokking  
        -   Adversirial robustness: Output resistance/invariance to perturbations in the input.  
        -   Delayed robustness: Grokkiong/generalization on adversarial samples (so this seems like Gorkking premium basically)  
        -   Piecewise function: A function that is partitioned into several intervals on which the function may be defined differently.
        -   circuit: A subgraph of the network(a combination of activated neurons) (This actually comes from another paper but it's reused here.).  
            Such circuit can be simplified, when using a continuous piecewise linear activation functions sucha as relu, to a single affine transformation.  
        -   Partition/region: Shape in the input space of a circuit.  
        -   Local Complexity: A measure of the number of different output partition/regions/splines in neighborhood in the input.  
            I will explain it more clearly later.  
        -   circular Matrix: A square matrix in which all rows are composed of the same elements.  
            Each row is rotated one element to the right relative to the preceding row.
        -   Training interpolation: Point at which the model as reached very low training error/loss (usually when it has overfitted?).  
    -   All DNNs architectures are a form of MLP, that is true for CNNs and Transformers.     
        I assume it's also true for other less well known architectures but this doesn't really matter anyway.    
        In fact, a conv layer is a specifc case of a simble linear layer, it's a circular Matrix.  
    -   All DNNs are continuous piecewise linear functions/continuous piecewise affine splines.  
        Yes, the paper uses different expressions for (what seems to be) the same thing.  
    -   The density of these pieces/partition/regions in a subregion of the input space can be expressed as local complexity(LC).  
        The LC of a point (like a training/test/validation sample) expresses:
        -   how many regions there are close to it
        -   how "non linear" the output is in that input region (ssuming that the regions have diffrent linear functions).  
        -   how complex the output is in that input region
        It is presented as a new progress measure on an equal foot with loss, accuracy ect.
        To measure the LC of a layer in the neighbood of a point:
        1.  P points are projected on a sphere of radius r centerd on the point.
            This composes a cross-polytopal.  
        1.  We pass the points/vertices through the linear layer.  
        1.  Vertices from differnt sides of the neurons hyperplanes will have different signs.  
        1.  Vertices with different signs than the one they are connected to by edges of the cross-polytopal increase the LC measure.  
        1.  Then we pass the points through the layers up to the next linear layer.   
        1.  Repeat until we reach the end of the model.  
            I believe that the originally convex Polytope will potentially loose over time its convexity.  
        It's still unclear to me tho by exactly how much does the LC increases by sign changes.  
    -   The training or test LC is the LC of the training and testing points respectively.  
        When they increase, it means that the number of linear regions/complexity/non-linearity around their respective points increases.
        Simillarly, when they decrease, it means that the number of linear regions/complexity/non-linearity around their respective points decreases.
    -   During training the training LC follows this dynamic: 
        1.  A descent, this one "is subject to the network parameterization as well as initialization"(extraact from the paper) and may not always occur.  
        1.  An ubiquitous(always happens) ascent that lasts until trainging interpolation.  
            ```
            The training LC may be higher for training points than for test points.
            Indicating an accumulation of non-linearities around training data compared to test data.
            ```
            Sligthly modified extract from the paper.
        1.  A second (if there was a first) descent of training AND test points.  
            The linear regions migrate away from training points, this is kinda obvious since training LC decreases.  
            During this phase, the authors have discovered  a less obvious phenomenom through spline cam:  
            `The regions tend to migrate away from the training+testing points and toward the decision boundry.`  
            **This is when the grokking happens.**  
            Interstingly enough, this happens before generalization can be recognized through train/test loss/accuracy.  
    -   According to section four here is how grokking is affected to hyperparameters:
        -   Parameterization(number of parameters): 
            Increasing parametrization whether through widdening or deepning,
            "hastens region migration, therefore makes grokking happen earlier".
        -   Weight decay: 
            ```
            Weight decay does not seem to have a monotonic behavior as it can (added the can for clarity) both
            delays and hastens region migration, based on the amount of weight decay.
            ```
        -   Batch Normalization: Always blocks grokking (I tried reading the Appendix to understand why but I got lazy).
        -   Activation functions: Was only tested on Relu And Gelu and both lead to grokking.  
        -   Dataset size: 
            Scalling good generizable training data hastens grokking.  
            On the other hand, scalling training set tha contains data that needs to be morized, slows down grokking.  
            This is demostrated in the paper by training an MLP on MNIST and randomizing a defined fraction of the training sample labels.  
    -   The paper also makes a connection between spline and circuits theory:  
        -   Each partition is a circuit.  
        -   Moving from one adjacent partition to the other corresponds to turning on or off a single neuron.  

27/05/2025:
-   Now I will try to use first the [spline cam code](https://github.com/AhmedImtiazPrio/splinecam) and then the [local complexity code](https://github.com/AhmedImtiazPrio/grok-adversarial/).  
    Turns out that spline cam does not work for skip connections nor batch norm layers so this is a bummer...  
    In the paper they show that they used a grid of local complexity measures to aproximate the results of spline cam.  
    I might use that instead.  

28/05/2025:
-   It took me a while to get the `train_resnet18_cifar10.py` demo github code of the paper working.  
    First because of the conda env (as always...) and then because it simply took ~2.5 hours to run the experiment on a V100.  
    But it did work eventually.  
    The thing is that the model only gets 70% test accuracy.  
    This is disappointing because:
    - wouldn't really call that "delayed generalization", maybe just "delayed 70%".  
    - the resnet from the kaggle notebook achieves 90% accuracy in 2 mins...  
    - the paper states that batch norm prevents Grokking for happening.  
    - The paper is called "grokking always happens and here is why"... but not when batch norm are used but batch norm layers are almost always used.
    

29/05/2025:
-   I took another look at the demo code and it turns out that while the test accuracy stagneted at 70%, the adversirial test accuracy reached 70% just at the end of training.  
    It looked like it could have kept going up (judging only by the adversarial test accuracy over iterations curve).  
    So now I am starting to wander if I actually should make the trainging last longer...  
    I guess I should also see if the "kaggle resnet" can get a similar score on adversarial test samples.  
-   I also think I should look into some videos/blogs/papers on adversarial attacks, I saw links on them in the superposition blog of Olah/Anthropic.  
-   I read a [reddit post](https://www.reddit.com/r/MachineLearning/comments/1defvmv/d_is_grokking_solved/) and its comment section.  
    In it, a scientist (if he is not lying saying that he presented a paper at ICLR) [said that grokking is over hyped](https://www.reddit.com/r/MachineLearning/comments/1defvmv/comment/l8d45rs/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).  
    It's just a delayed generalization which will always be worse than a "normal" quicker generalization.  
    It says that grokking mostly happens in accademic/edge cases which contracdicts what the paper says(tho the comment was posted later)...  
    He also says that grokking can be induced or ablated using a single output scaling hyperparameter.  
    So I guess i could verify that?..
-   I think my next steps are going to be:
    1.  Learn a bit more about adversarial robustness.  
    1.  See if the "kaggle notebook" performs well against adversarial samples.  
    1.  Monitor local complexity on it.
-   I read about Fast Gradient Sign Method (FGSM), it's such an elegant way of finding the best attack.  
    Before reading about it I thought it would consist in searching the closest descision boundry but no.   
    The adverserial training however is a bit less elegant. I found it representative of the "empirical state" that the deep learning field is in.  
    (Let me get down of my high horses ... thanks, let's continue)
-   I found this [kaggle notebook] (https://www.kaggle.com/code/seasonxc/fgsm-pgd-and-adversarial-training-cifar-10#What-is-FGSM?).
    It gives an overview of what PGD is, how to incorporate attacks in training.  
-   Looking at the attacks, I start to understand why the paper code resnet had such a low accuracy. 
-   Tomorrow I will try to implement the FGSM and PGD attacks.

30/05/2025:
-   Found this [nice PDF](https://files.sri.inf.ethz.ch/website/teaching/riai2020/materials/lectures/LECTURE3_ATTACKS.pdf) on PGD to understand it a little better than the simple overview given in the adversarial training kaggle notebook.
-   Wrote the first version of PGD.

02/06/2025:
-   Did a little bit of refacto to test the adversarial attacks in another notebook than the one I use to train the model.
-   Created the first adversarial attacks in a new notebook, seems to work pretty well.  
    I used the same PGD params as the ones in the "NNs always grok" code.
    Then I tested the accuracy to the model from 1 to 10 iterations.
    The model fails at 5 iterations.
-   I tried to add the PGD attacks into the training but I get a `RuntimeError: Tensor 0 has no grad_fn` error...
    I'll try to make work tomorrow.  

03/06/2025:
-   Turns out I had this error because the `record_metrics` of Trainer class had the `torch.no_grad` call as a decorator.    
    I wish the error was something like "can't call Tensor.bacward because no_grad is active" but that's okay...  
    I simply to call it as a context manager only for the forward call of the model and that fixed it.  
-   Without any change to the other hyperparameters I get 55% on the adversarial attacks on the test set and now the training takes ~15 mins.  
    Granted this is less than the "NNs always gok" code but I haven't changed any hyper parameter AND it takes ~15 mins which is a lot faster.  
    By simply increasing the number of epochs from 8 to 24 I get a 66% and the training now takes.  
-   I would need to update the max learning rate. To do so I would need to implement an learning rate finder as described in a blog I read a while ago.  
-   Before tho I would need to speed up the generation of attacks because it's taking way too much time.  
-   I asked uncleGPT what are the most commonly used adversarial attacks and one of them caught my attention: free Adversarial Training (freeAT for short).  
    I read its paper abstract and it seems very elegant, and efficient.  
    I will definitely try to implement this tomorrow.  
-   The paper mentioned Wide Residual Net (WRN for short) I took a look at a video about it not sure I will use that tho.

04/06/2025:
-   Worked on the video editing/creation app for Dominique.

10/06/2025 -> 12/06/2025:
-   Worked on the video editing/creation app for Dominique.
-   Finished(?) the app!
-   It took me more time than expected but this will be a nice addition to my portfolio/resume/CV.

13/06/2025:
-   Familiarized myself with the very basics of LeRobot datasets and overall LeRobot library(I'm doing a LeRobot hackthon this weekend).

14/06/2025 - 15/06/2025:
-   Participated to a LeRobot hackathon that was createad to popularize the new [so101 robotic arm](https://youtu.be/vC7E6ZmXBT8?si=_AvzT4KkwfuXpAll).  
-   I teamed up with Safwan Ahmad (awesome guy).  
    He already had two setup/working (robotic) arms, one leader and one follower.  
-   I already had a Vast workflow set up so we naturally split the work in half:  
    -   I would take care of model training.  
    -   He would handle all the code closely related to the robotic arms: recording our datasets and running the model on the arms.  
-   Our goal for the hackathon was to first train a model that would sort small ~0.5cm cubes to set up a pipeline to then implement more complex tasks.  
    Turns out that sorting these cubes was actually a very diffcult task.  
	The main issue is that the cubes were very small and the robotic claw is also pretty small.
	Watching at the wrist camera feedback we realized that it was actually pretty difficult (even for us) to see if the cube was or wasn't in the right position to grab the cube.  
	Also we realized that most of the models that we trained were actually overfitted.  
	We realized it late because didn't make a test/train split (whivh is a bad idea I know) because we had recorded a fairly small dataset and because there were "test" argument in the provided script to train the model.  
	Maybe we just had to do some manipulation of the hugging face dataset instead of providing some train-test-split argument to the training script.
	What happend is that we tested a [model checkpoint that we thaught was underfitted](https://huggingface.co/Mauro-Abidal-Carrer/4k_steps_smovla_new_dataset)(4k training steps) while the training was finishing.  
	And only when we tested the last [checkpoint of the model](https://huggingface.co/Mauro-Abidal-Carrer/15k_steps_smolVLA_new_dataset)(15k training steps) did we realize that 15k steps was too much.
	The aforementioned model was a finetuned smolVLA model.
	We also tried to train an act model but it didn't work.
-	We had two datasets:
	1. A [dataset](https://huggingface.co/datasets/SafwanAhmad/smol_test_safwan_odd_one_out) where the claw was held vertically while grabbing the cube.  
 	2. A [dataset](https://huggingface.co/datasets/SafwanAhmad/smol_test_safwan_odd_one_out_camTop) where the claw was held horizontally while grabbing the cube.  
  	3. Safwan also recorded a dataset but the recording conditions were not good enough it seems + I had trained an act model isntead of a smolVLA model.    
-	Overall the best performing model was a 4k training step model trained on the second dataset.  
-	We wanted to try to add a second wrist camera on the other side of the claw we thought that it would enable the model to clearly see if the object was in between the claws.  
	Unfortunatly we ran out of time before we could try that.  
	That's mostly my fault tbh, I should have arrived earlier...  

16/06/2025:
-	I looked on the internet for cosine annealing+warm up(which one of the most popular lr policy for large models) but I din't find any usefull insights.  
-	I restarted watching the TuneAdorable video on warmup.  
	It goes over a part of the paper that says that some model will naturally decrease the learning rate sharpness as they also increase their loss over training steps.  
	While some other models tend to have increase in lost sharpness over training steps.  
	This reminded me of the double descent of local complexity for some model and single ascent+descent for other models in the NN always grok paper.  
	I wander if there is a link between the two.

17/06/2025:
-	What if I used the loss sharpness as a regularization loss itself?  
	I'm starting to link loss sharpness, local complexity and adversarial attacks together.  
	They seem to be kind of measuring the same thing:  
	**Concepts**:  
 	-	Loss sharpness is a measure of how much the loss changes over model param changes.
 	-	Adversarial attacks are the highest loss value points in a given range around a sample.
  	-	Local complexity is a measurea of how much the output changes around a sample.
   		(Btw I am ~90% sure that adversarial attacks would make up for better probes for local complexity measures instead of random orthognal points around a ball.)
	**Properties**:  
	For a given loss score:  
	-	The lower the loss sharpness the better generalization is.
 	-	The lower the loss on adversarial attacks (in a given range ofc) the better the generalization is.
	-	The lower the local complexity, the better generalization is.  
   	The current ways (that I know of) to satisfy these regularization requirements are:  
  	-	l1/2 norm
   	-	data augmentation
	-	adversarial training
 	-	dropout layers
    I consider the last three methods to fall in the same family of regularization techniques: add some noise and force the model to have the same output.  
   	That's because changing the input during training and expecting a similar output is equivalent to having a lower loss sharpness because it would be the same as changing the parameters and expcting a similar output.  
   	*THis sentence probably won't make sense to me neither in a few days but it does for me right now.*  
	While the last three are battle tested, I feel like there should be an aproach that is analytcal based rather than empirical based (the adv training is kinda analytical and empircal since it doe suse the loss gradients).  
    > Note: I don't count architecture modifications such as layer/batch/sample normalization or skip connections since they are architectural modifications.  

	Hopefully there is some sort of math trick to enfore loss flatness.  
	Maybe something like second degree gradient or l1/2 norm on the gradients themself?  
-	I started a new branch on my learning-deep-learning repo.  
	To test my theory I will benchmark a small MNIST CNN (to get results ASAP) and compare all the commonly used regularization methods + mine.  
	Optionally, I will use some interpretation tools but I doupt I will have enough time for that.  
-	Everything is setup I've got the model trainging done + test with adv attacked samples to go from ~98% accuracy to ~60%.  
	Additionally, I could reduce the number of training samples.  
 
18/06/2025:
-	I asked chatGPT if my idea of enforcing loss fltaness was a good idea and if there are potential flaws.  
	It said that it's a pretty good idea (as it always does lol) and that there are already are reg methods that have been suggested such as Sharpness Aware Minimization (SAM).  
	That's exactly what I was thinking about!  
	Its paper is from 2020 and it doesn't seem like it has been actually used that much, it's the first I come across it(tbf I haven't checked out the training mehtod of every model but you get what I mean(right?)).  
-	I watched [this introductory video](https://www.youtube.com/watch?v=k6-jJ58MFKU&ab_channel=Yuxiang%22Shawn%22Wang) on it.  
	And the big (and only?) difference between my idea and the papaers implementation is that the loss sharpness is measured through a you guessed ... probe ball of the parameters.  
	Basically it compares the loss diff between the current model and a randomly modified one.  
-	Hopefully I can improve this by using some sort of math trick like the ones I wrote about yesterday.  
-	Now the plans have changed, I will first test SAM with a lot of probes, since I am training a small model on a small, (too?)easy dataset.  
  	The idea is that an empirical method with a lot of compute is as good as analitycal one.  
 	If it doesn't perform correctly then it wouldn't even be worth experiment on an anlitycal version.  
-	I probably would have to make all of these assumptions checked by chatGPT before spending too much time on this.  
-	I checked with uncleGPT and.... the analytical SAM versions/alternative have already been proposed, in fact some of them were proposed before SAM.   
	Basically, the analitycal way of measuring the loss sharpness is by using the Hessian of the model params.  
-	I watched this [nice video about the hessian](https://www.youtube.com/watch?v=5qD53Exg6kQ&ab_channel=DigitalMedia-ImperialCollegeLondon).
-	I read this [nice blog post about the hessian](https://maximiliandu.com/course_notes/Optimization/Optimization/Notes/Hessian%20Matrix%20b8813e511a1745bdaeaaf71e758eac5d.html).
-	I read the (very) beggining of its [wikipidia page](https://en.wikipedia.org/wiki/Hessian_matrix).
-	KTs:
	-	The hessian is the partial derivative, of the partial derivative of the model, i.e: it's the second degree partial derivative of the model.     
	-	It allows us to locally aproximate the loss as a quadratic function.    
		And from that we can derive a bunch of things like, are we in a local minimum/maximum/saddle point (I still don't fully understand the latter).  
	-	I should probably read more /watch more videos about it.  
 -	uncleGPT also said that the eigenvalues of the Hessian are sometimes used.  
 	I defenetly need to understand the eignevalues better tho, they seem pretty cool(this will probably lead me down a rabbit hole).  
-	Given that the other analytical versions are not used either I starting to think that it's not really a practical solution to use the loss sharpness as reg loss.  
-	I think I will keep looking into it until the 20th this way I will improve my understanding of the math behind it all.
-	Watched this [video on eigen values](https://www.youtube.com/watch?v=1sDBruay100&t=3s&ab_channel=BrainStationAdvanced).
	Kts:
  	-	Eigen vectors of a matrix point toward the directions where the "dotted" vectors do not rotate/change direction.
   	-	Eigen values tell how much "dotted" vectors are strched if they are on the corresponding eigeven vector's line.
    -	If eigen values are imaginary then "dotted" vectors always rotate.  


19/06/2025:
-	I watched this [video on ViT models outperforming resnet when trained with SAM](https://youtu.be/oDtcobGQ7xU?si=LukXniRNHg6Ag054).
	Kts:
	-	I missunderstood how SAM works, it doens't actually randomly probe the model parameter's space but rather adds the (positive) gradient to the model and then remasures the loss.  
 		Then i believe it uses the diff between the current model's loss and the "adv model"'s loss as an additional reg loss.    
 		This is similar to adv attacks but instead of adv changing the model's input we change it's params.  
  	-	I wonder if this is more efficient than adv training and if it's possible to implement some sort of "freeSam" that would be analogous to freeAT.  
   	-	As I understood the video, it seems that resnets don't benefit as much as the ViT from SAM since they already have regullarization inforced in their architecture.  
	I'm pretty happy actually that the idea I had turned out to have already been tested AND yelled positive results.  
	I interpret that finding as a sign that I understand what I am studying which is pretty valuable since I am studying on my own without someone giving me signs on what do right or wrong.  
-	I also watched a video of the same author on [how he got into deepmind](https://youtu.be/SgaN-4po_cA?si=LS7tRukI28HlLw1t).  
	KTs:
	-	Don't focus on one company at a time try multiple ones.
 	-	The success rate is super low.
  	-	You need a referal.
   	-	You can get a referal by creating DL content and getting attention from a employe in a company
   	-	You can get a referal by contributing to open source code (sounds more feasable).
   	-	You need to know software theoretical stuff mostly data structures+algorithms.   
    	It seems like competitive coding is a good way to learn that + there are some books made for that like "cracking the coding interview".
	-	He also read the papers  published (and more generally work I asume) of people he would get interviewd by, prior to those interviews.
 	Given the expected low success rate I should probably look into more job posts to find more job expectation/requirements.
-	I have only one day and half left from my deadline of my goal to learn and contribute to fundemental/academic work.
	I feel like I *could* do something in a ~week but will most likely be useless and also lead me down a rabbit hole.
	I think it would be better for me to either participate to the [NeurIPS kaggle competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview) and/or the [Geophysical Waveform Inversion](https://www.kaggle.com/competitions/waveform-inversion).  
	The second one is finishing in 11 days.  
	I could pick up already made code get familiar with the excercise of doing a competition and then restart fresh on the NeurIPS one, finishing in three months.  
  	Danm  three months is actually a lot...  

20/06/2025:
-	Okay now I am fully committed to doing those two competitions.
	It took me a while to understand what the competition is about but it turns out that it's actually fairly simple:
	-	The goal of the competition is to infer what lies beneath the ground.
   	-	To do so we use sound waves emmited from a source (like a thumper truck or a detonation).  
    	Then we record the sound for a few seconds with some microphones arranged along a line and we repeat the experiment (each called "shots" I believe).
	-	The resulting data is a 3D array of shape [nb shots, duration * record frequency, nb microphones] where the value is the magnitude of the sound wave (hopefully I didn't miss use that term).
	-	We can use that data to infer a [velocity map](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FFigure-a-summarizes-the-DL-FWI-information-flow-Figure-b-shows-the-SEAM-velocity_fig1_366026572&psig=AOvVaw3qghZDhXOhcRGGkR9584Fi&ust=1750775715475000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCJDKi_fhh44DFQAAAAAdAAAAABA1) of shape [width, height] where the values are in some speed metric like km/s.  
-	I will need to crank in more hours a week right now I must be at around ~27 hours a week.  
  	My goal is to get a entry level job as DL engineer.  
	To get to the required level and have a proper CV/portfolio I will probably need to get up to ~40 hours a week.  
-	I looked at the most voted notebooks and I found this [ConvNeXt solution](https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline).
	It refers to a [HGNet solution](https://www.kaggle.com/code/brendanartley/hgnet-v2-starter), which itself refres to this [Unet solution](https://www.kaggle.com/code/egortrushin/gwi-unet-with-float16-dataset).
	I also found this [Physics informed + transfer learning Unet paper](https://arxiv.org/pdf/2212.02338).
-	I started reading the paper, it's only 5 pages.
	It feels like a breath of fresh air compared to the other verbose+repetitive+esoteric papers I have read so far.
	However it does also feel like some parts of it have been literally cut out...

22/06/2025:  
-	Finished reading the paper, KTs:  
	-	The paper links to deep wave which is a full wave form inversion package built on top of Pytorch to do Physics informed training, thats really cool.  
	-	The transfer learning and PINN are great...
 	-	They used 2xV100 which is *fairly* affordable on Vast.
	-	They removed the fist encoder to decoder skip connection to prevent recording arctifacts to impact the quality of the results.  
  		I believe there's also some other stuff done to the skip connections to prevent the since the vertical vertical dimensions of the input and output don't really correspond.  
  	-	I believe there is some pretraining stuff I haven't quite understood but I will get back to it if I chose to follow there method.
-	I read the Unet solution notebook, it uses a 16float (instead of 32float) version of the dataset.  
	Downloading just a subset of the 16 float dataset takes 1h30mins. That's gonna be an issue given that I tend to destry my instances to save on billing at the end of the day.
	If I stop then restart them I have to hope that someone hasn't rented the GPU in the meantime which just as inpractical as recreating a new instance.

23/06/2025:  
-	I read the HGNet solution, KTs:
	-	ngl the architecture it's pretty damn complex.  
		I skip some architectural details to save some time as this not even the best solution of the author.    
	-	It uses a [backbone trained on Imagenet](https://huggingface.co/timm/hgnetv2_b2.ssld_stage2_ft_in1k) which is super weird since that's a 3 channels input when FWI has only one channel and has very different shapes to photos.    
		If I didn't miss interpreted the origin of the backbone, that seems like a huge room for improvement(to not say terrible idea).    
		I would probably use the same backbone used in the paper instead. 
-	Turns out the author as a better less voted (compared to its ResneXt submission) submission that uses CAFormer (wtf is that architecture now (-_-)?).
	it does refer to the ResNeXt notebook tho so will still have to read that before.  
