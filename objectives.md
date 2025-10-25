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
    - Plastic dismentaling?
- kaggle
    - Child Mind Institute 2025 competition (done, 148th out of 2657 participants, bronze medal, finished in 2nd of September)

### Andrej Karpathy's GPT2 tutorial (16 September - 7 October)
- Learn how to get the most out of the GPUs, what's the best GPU for the task.
- Learned how transformers and their tokenizers work.
- waisted a week on dum typos "hidden" in the model definition.
- Learned about Distributed Data Parralle pytorch package.
- Learned a bit more about W&B.

Applied to MATS fellowship the 2nd of October

Apply to LASR and astra fellowships.
- Clean the transformers and CMI repos
- Clean the transformer W&B workspace
- Make the transformer beat GPT-3 in Hellaswag eval?
- Fix transformer data loader?
- Update the webpage?
  
- Open source contribution  
  *This should allow me to validate the "open source contribution", "worked with LLMs"(if I do something with them ofc), "done some interpretability work".*
  

### Get a job!!
- Relancer MATS(send)
- demander a Martin about Lixo(send)
- Send spontanous participation to Doctolib(send)
- Send CV to MATS Program (done)
- Send CV to goodfire for fellow ship position
    - Implement SAE
        - SAE on uncensored LLM + steer llm to safety
            - Create dataset of activations on harmfull questino/answers
                - input/output tokens, residual streams activations, question category
            - Train SAE
                - p annealing ?
            - Eval SAE
                - calssify question/answer category with linear+softmax
                - l0 loss eval
                - eval LLM with reconstructed residual activations
                    - eval LLM performance
                        - Hella swag like eval
                        - sample random question/answers
                    - eval LLM harmfullness
                        - Hella swag like eval
                        - sample random question/answers
            - Steer LLM
                - Generate safe answers to harmfull questions
                - Found harnful/compliant answers SAE features
                - Found safe/non compliant answers SAE features
                - Create harmfull loss: SAE activations @ harmfull vector - SAE activation -> SAE activations @ (harmfull vec - safe vector)
                - back prop loss to residual activation
                - eval again
                - that's "it"?

<!--        - model: -->
<!--        - preprocessing: -->
<!--        - training: -->
<!--            -  Switch back to one cycle lr scheduler -->
<!--            -  Equalize target distribution through data augmentation -->
<!--            -  Use free adversarial training? -->
<!--            -  Use EMA of model -->
<!--            -  Focal Loss -->
<!--      - meta: -->
<!--            -  use with top public score notebook   -->
<!--            -  read top public score notebook to try and recreate the weight search process. -->
<!--            -  Search for other preprocessing steps. -->
<!-- -->
<!-- -->
<!--### Submission: -->
<!--- merge with best lb score notebook -->
<!--- preds with noise -->
<!---   mixup ratio # -->
<!--  + dropouts in head # -->
<!--  + Focal Loss  -->
<!--  + more cross axis energy ? -->
<!--  + more_folds -->
<!--  + EMA of model -->
<!--  + hp search space: -->
<!--    - dropout and gaussian noise -->
<!--    - mixup alpha and ratio -->
<!--    - focal loss -->
<!--    - ema params -->

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
