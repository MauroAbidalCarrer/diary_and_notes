### Skills:
  - Programming languages: Python, C, C#
  - Programming Frameworks: PyTorch, Pandas, Numpy, Plotly, Optuna, scikit-learn
  - Spoken Languages: English, French, Italian

### Experience and projects:

- **Child-Mind-Institute-2025 Kaggle Competition**  
    - reached **146th place out of 2600** participants with an **0.83 F1 score** as a **solo competitor**.  
    -   Data preprocessing using pandas, numpy and sklearn.
    -   Cross validation training parallelized over mutliple GPUs using pytorch's multiprocessing module.  
        The parallelization enabled to scale/speed up to 10x.  
    -   Hyper parameter tuning using **Optuna** and the afford mentioned training solution.
    -   **Independently researched Multi-Task Learning** and implemented it to better leverage the dataset.
    -   This in part enabled me to outperform the best public notebook of the competition with a 0.81 F1 score.  

- **BIB Batteries – Data Science Intern (1 year)**  
    - Predicted **State of Health (SoH)** of EV batteries using real-world fleet data (+14k vehicles, mostly Teslas).  
    - Learned to write clear and reproduceable notebooks.
    - Designed, delivered to **production** and maintained a **medallion architecture ETL+ML pipeline**:
        - Converted raw per-manufacturer JSON to parquet files.  
        - Preprocessed vehicles time-series (charging states, speed, current properties, temperature, etc.).  
        - Estimated SoH with a novel approach later validated against physical tests with **1% MAE**(SoH is estimated in percentage).
        - Aggregated fleet-level data to provide business insights, e.g:
            - SoH degradation per mileage per model
            - Principal influnecial factors of SoH loss
            - resale value predictions
    - Stored intermediate data as parquet files in **S3 bucket** and business insights into a **Postgre SQL** database.

- **Personal Learning**  
    - Implemented more CV concepts (ResNet, FreeAT augmentations) using Pytorch.  
    - Built a toy CNN library from scratch using NumPy.  
    - Completed *Neural Networks from Scratch* book for fundamentals in deep learning.  

- **Freelance Unity Game Developer (6 months)**  
    - Delivered small game projects for clients.  
    - Worked across gameplay programming, prototyping, and asset integration.  

-  **School 42 – Data science branch**
    -   Motor imagery classification task from (EEG) recordings(i.e recognizing movements from brain-waves).
        -   Used MNE library to preprocess to apply band pass filtering and segment experiments.
        -   Learned the statistical concepts behind dimensionality reduction algorithm "Common Spatial Pattern" (CSP).
        -   Implemented a cross validation training pipeline  
        -   Modified the pipeline to perform **real time inference**.

- **School 42 – Main Cursus**  
    -   Learned low level software engineering concepts through a projects based program.  
        Note: Not all the projects are mentioned here.
        - **Won Vivatech game jam**, with project showcased at Vivatech.  
        - **webserv**: Nginx-like web server in C++.  
        - **minishell**: Bash reimplementation in C.  
        - **philosophers**: Solved dining philosophers problem in mutli threading.
        - **MiniRT**: Basic ray tracing engine.  
        - **FDF**: Mesh rendering in orthographic & perspective projections.  
