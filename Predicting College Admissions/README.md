## Predicting College Admissions

### Abstract
We all have been in the situation where we are searching for colleges to apply for. These days, there are a lot of colleges around and each of them has a different acceptance criterion for admissions. While filling out our application we want to make sure that we choose a college with a higher probability of accepting our application - one, because the application costs are high and second, because we want to apply to a college which admits in the range of our test scores.

In this project, I use *Supervised Machine Learning* to predict the number of admissions in a college based on the number of applicants providing test scores (SAT/ACT) and the scores itself.

### Outline

- Create a Regression Model to predict the admissions rate in colleges, based on their accepted test scores (SAT/ACT).

- Multiple years of data from [Integrated Postsecondary Education Data System (IPEDS)](https://nces.ed.gov/ipeds/) was used for developing this model and the files can be found in the [data dir](./data)

- The code is located as a jupyter notebook [here](./work.ipynb)

- The model is developed in python using scikit-learn and keras with tensorflow. Seaborn, pandas, numpy, are some other python libraries used for data processing and visualizations.

- The final project report can be found at [report.pdf](./report.pdf)

- The weights for the trained Neural Network model can be found in [dl_weights dir](./dl_weights)

*This project was accepted as the capstone project for my Udacity Machine Learning Nanodegree.*


