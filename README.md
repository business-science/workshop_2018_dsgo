# DSGO 2018 Workshop - Business Science

___Get ready to learn how to predict credit defaults with `R` + `H2O`!___

## Program

<a href="https://www.kaggle.com/c/home-credit-default-risk">
<img src="00_images/kaggle_credit_default.png" style="width:30%;" class="pull-right">
</a>

- Data is Credit Loan Applications to a Bank. 

- Best Kagglers got 0.80 AUC with more 100's of manhours, feature engineering, combining more data sets 

- We'll get 0.75 AUC in 30 minutes of coding

<div class="clearfix"></div>


## Data

- Kaggle Competition: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

- Data is large (166MB unzipped, 308K rows, 122 columns)

- Will work with sampled data 20% to keep manageable


## Module 01 - H2O

The goal of Module 01 - H2O is to get you experience with:

1. The R programming language

2. `h2o` for machine learning

3. `lime` for feature explanation

4. `recipes` for preprocessing

---

## Installation Instructions

### Step 1: Install Docker CE

_Skip this step if you already have Docker Community Edition installed_

[Docker Community Edition Installation Instructions](https://store.docker.com/search?offering=community&type=edition)


### Step 2: Run the DSGO Workshop Docker Image

In a terminal / command line, run the following command to download and install the workshop container. This will take a few minutes to load. 

```
docker run -d -p 8787:8787 -e PASSWORD=rstudio -e ROOT=TRUE mdancho/workshop_2018_dsgo
```

### Step 3: Fire Up RStudio IDE in your Browser

Go into you favorite browser (I'll be using Chrome), and enter the following in the web address field.

```
localhost:8787
```

### Step 4: Log into RStudio Server

<a href="https://www.kaggle.com/c/home-credit-default-risk">
<img src="00_images/rstudio_server.png" style="width:30%;" class="pull-right">
</a>

Use the following credentials.

- __User Name:__ rstudio
- __Password:__ rstudio

<div class="clearfix"></div>


### Step 5: Load the Project From GitHub

_Wait for instructions from Matt._

The URL for the GitHub project is:

https://github.com/business-science/workshop_2018_dsgo


---

## Further Resources

- `tidyverse`: A meta-package for data wrangling and visualization. Loads `dplyr`, `ggplot2`, and a number of essential packages for working with data. Documentation: https://www.tidyverse.org/

- `recipes`: A preprocessing package that includes many standard preprocessing steps. Documentation: https://tidymodels.github.io/recipes/ 

- `h2o`: A high-performance machine learning library that is scalable and is optimized for perfromance. Documentation: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html 

    - GLM: Elastic Net (Generalized Linear Regression with L1 + L2 Regularization)
    
    - GBM: Gradient Boosted Machines (Tree-Based + Boosting)
    
    - Random Forest: Tree Based + Bagging
    
    - Deep Learning: Neural Network
    
    - Automated Machine Learning: Stacked Ensemble, All Models and Best of Family

- `lime`: A package for explaining black-box models. LIME Tutorial: https://www.business-science.io/business/2018/06/25/lime-local-feature-interpretation.html 

