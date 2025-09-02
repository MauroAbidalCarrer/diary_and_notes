### theoretical/fundamentals concepts: 20th of May to 1st of June and 11th of June to 20th of June (~20 days)
*Learn pytorch, the theory and hopefully work on top on a recent paper and make a new one(maybe just a blog?).*
Goals:
    - Be able to train a (SOTA?) model on a common task (CV, NLP, whatever).
    - Be able to explain:
        - WHY I used the given hyper parameters + architecture.
        - WHY the model works 
Tasks:
- read VGG + resnet papers (done)
- read resnet paper (done)
- read learning rate finder (done)
- read one cycle policy blog post (done)
- Reimplement 90% on CIFAR10 (done)
- good model on imagenet ?
- YOLO ?
- Transformer ?
- Gen image model ?
- read blog/paper/watch video on adversirial attacks(done, kind of...)
- Read Neural nets always grokk and here is why (done)
- (re)Read building block of interpretability blog ?
- Use adv attacks as better local complexity ?
- Read circuit theory ?
- Read spline theory ?
- Read spline cam ?

### Practical work: 20th of June to 20th Jully (30 days)
*Do some work on a practical,real world projects to put in my portfolio and improve my skills.*
- Reinforcement learning
    - chess?
    - sc2?
- huggin face's lerobot?
- kaggle
    - Child Mind Institute 2025 competition
        - model:
        - preprocessing:
        - training:
            -  Switch back to one cycle lr scheduler
            -  Equalize target distribution through data augmentation
            -  Use free adversarial training?
            -  Use EMA of model
            -  Focal Loss
      - meta:
            -  use with top public score notebook  
            -  read top public score notebook to try and recreate the weight search process.
            -  Search for other preprocessing steps.


### Submission:
- merge with best lb score notebook
- preds with noise
-   mixup ratio #
  + dropouts in head #
  + Focal Loss 
  + more cross axis energy
  + windowed cross axis energy
  + more_folds
  + EMA of model
  + hp search space:
    - dropout and gaussian noise,
    - mixup alpha and ratio
    - focal loss
    - ema params

<!-- llkh0a solution -->
<!-- - thm groups + more_cross_axis_features -->
<!-- - Use different branches for each tof sensor (we can probably use groups in the alexnets intead of using multiple alex nets) -->
<!-- - add gravity direction -->
<!-- - add cross axis energy see this [notebook](https://www.kaggle.com/code/wasupandceacar/lb-0-841-5fold-single-model-with-split-sensors) -->
<!-- - Move diff computing in model to reduce VRAM usage -->
<!--            -  speed up training by parallelizing folds training -->
<!--            -  meta data/performance EDA -->
<!--            - Turn demogrpahics into auxiliary targets -->
<!-- - phase during the sequence "behavior" column -->
<!-- -  Aggregate patches of the ToF sensors data -->
<!-- -  Unify preprocessing and training/inference notebooks into a single one to avoid waiting for zip, upload, kaggle processing and downlod delays. -->
<!-- - Increase the number of rnn layers to 2. -->
<!-- - Use 100% percentile for sequence len padding -->
<!-- -  Collapse non-BFRBs target into a single one to ease learning -->
<!-- - sequence wise std norm -->
<!-- -  put std normalization step in the model to since we are using CV model ensemble -->
<!-- -  Update hyperparams (again): -->
<!--    -  Use smaller btach sizes, top notebooks use 64 batch size, I use 256 -->
<!--    -  Use a lot more epochs, top notebooks use ~100 epochs where I only use ~25 -->
<!--    -  Increase patience, top notebooks use 40 patience -->
<!-- Use post/pre truncating/padding instead of center truncating/padding -->
<!-- -  use third branch for thm input -->
<!-- - use other paddig methods like "same" or "reflect" padding for convolutions and sequence padding -->
