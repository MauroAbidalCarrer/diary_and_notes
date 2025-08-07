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
- kaggle  
    - Child Mind Institute 2025 competition
        -  Use EMA of model
        -  Try to avoid overfitting by taking the best mean CV score across all folds instead of using the peak CV epoch of each folds.
        - Aggregate patches of the ToF sensors data
        -  EDA input meta dataand model performance
            -  true seq length
            -  nan ratios
            -  thm/tof removed?
            -  demographics?
            -  target class
            -  compute pearson correlation?
            -  compute recall and precision to see which one is preventing the f1 score to increase
            -  orientation
            -  phases durations
        -  make hyper paramater tuning more efficient
        -  speed up training with implementation improvements?
        -  speed up training by parallelizing folds training
        -  ensemble:
            -  Use more models
            -  Weight models
            -  Both solutions above could be powered by the EDA
        - Equalize target distribution through data augmentation
        - Augment training by using external datasets and self supervised training on them.
        - Use Self supervised training:
            - Improve embending with masking + extra decoder
            - Use more of the dataset data:
                - extra data:
                    - orientation of the subject during the sequence
                    - phase during the sequence "behavior" column
                - By creating new targets during training (only viable for orientation column)  
                  Targets would be a cartesian product of the competition's targets adn the orientations.  
                  - During training: the model is trained on those clases
                  - During inference: the model's output is collapsed to only the competition's target class

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
          
- Reinforcement learning
    - chess?
    - sc2?
    - huggin face's lerobot?
