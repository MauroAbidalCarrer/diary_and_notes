Please compress the following experience into a resume structured way.    
Please write you answer in a code block so I can copy paste it.  
In chronological order, here is my projects/experiences that I would like to compress:   

Child-Mind-Institute-2025 Kaggle competition:
-  reached **146th place out of 2600** participants with an **0.831 F1 score** as a **solo participant**.
    -  Data preprocessing using pandas, numpy and sklearn
    -  parallelized folds training over mutliple GPUs for Cross validation using pytorch's multiprocessing module.
    -  Hyper parameter tuning using Optuna and the afford mentioned training solution.
I would like the reader to take away that I:
- learned to do reasearch entirely on my own.    
- sped up cross validation training by up to 10x (when renting 10 GPUs instances).  
- That I got way above the best public notebook, I didn't just copy paste the best public notebook and got 146th I made my own solution.

Learned pytorch and implemented more advanced concepts:
- architectures like resnet(implemented), resnext(not implemented).
- Augmentation methods like freeAT(implemented ).  

Made my own toy CNN library from scratch using Numpy.
Learned the very basics of CV architectures.

I did a 1 year internship at BIB Batteries.  
I was tasked with predicting the SoH of the EVs batteries based on their data:  
- time series variables: of SoC, speed, charger type (when charging), charger electric current properties(voltage, amps, DC/AC) (when charging), temperatures, ect...  
- Static variables: model, battery capacity, default range, ect...  
The advantage over estimating the SoH with a specialized charger is that we can estimate it in real time and at scale since we don't need to force the drivers to perform tests.  
Another dev was tasked with performing the requests to the constructors API (Tesla, BMW, ect...) to get the data and stored the answers as raw json into an s3 bucket.  
We were monitoring 14k cars from multiple car leasers 13k of which were Teslas.  
We were pulling data from those cars daily.  
I implemented a pandas+sklearn pipeline with a medliation architecture with 4 steps:  
-   convert per manufacturor/vehicle/request json files into a per manufacturor parquet file without any preprocessing. (bronze)  
-   Preprocess time series: Segment the time series by computing in_charge, is_moving masks. Set the correct dtypes, convert to the same metrics. (silver)  
-   compute SoH (gold pt.1)  
    My SoH estimation were compared with the results of physical tests and ended up having a (very low) 1% MAE.  
    These estimations were stored in a postgre SQL data base.
-   Use fleet wise data in conjuction with the estimated SoH to infer more valuable business data such as per model degradation of the SoH per mileage.  
    These insights were not publicly known at the time.  
    They also had the potential to better inform the second hands buyers of those cars.  
    Some models were better suited to be rebought since they had better per mileage SoH loss.  
Each step was cached as parquet files in a separate folder in the S3.
I would like the reader to take away that I learned to do reasearch independantly while incorporating feedbacks.   
By writting clean reproduceable notebooks.  
Estimation of the SoH in this manner is not publicly documented my superior/manager was very busy so we "only" had meeting every two ~days.  
The rest of the time it was up to me to research novel ways or better ways to accomplish that mission.  
I would also like that the reader takes away the fact that I put the very solutions I reaserched into production and maitained the pipeline, and that I had to make the code efficient because it was running on a "on budget" machine.
I would also like that the reader takes away that this is a fast moving startup where I had to juggle with multiple tasks and theirm respective deadlines.


Did my school's data scince branch including a motor imagery classification project based on EEG recordings.  
Used MNE, numpy and sklearn
my pipeline was:
-   low/high pass band filtering
-   my implementatio of Common Spatial Patterns 
-   Lasso classification
I would like the reader to take away that I learned the basics of machine learning/datascience including the math (I learned how CSP works)

Made a contribution to a github repo "open code interpreter" open source version of OpenAI's code interpreter.
I added the option to searilaze the user, assistant's message, code and code output into a single notebook file.

I followed the Neural networks from scratch book.

Did 6 months of Unity game dev freelance

Won a team game jam organized by Vivatech as a side project in school, the game then got presented at Vivatech.
I would like the user to take away that I know how to work in a team under tight (2 days in this case) deadlines.

Did the 42 school main cursus, biggest projects include:
- webserv, ngnix clone in CPP
- MiniRT basic ray tracing engine
- FDF: rendering of simple meshes as wireframes, in orthographic and perspective projections
- minishell: reimplemented bash in C
I would like the reaeder to take away that:
- I know most of the low level OS concepts that most dev don't know about:
    - what is a
        - process 
        - the notion of parent/child processes
        - signal, pipe, socket
        - ect...
- That I am rigorous in my code.


Thanks in advance for your andwer!