# MACHINE LEARNING ----

# Objectives:
#   Size the problem
#   Prepare the data for Binary Classification
#   Build models with H2O: GLM, GBM, RF
#   Inspect Features with LIME

# Estimated time: 2-3 hours



# 1.0 LIBRARIES ----
library(tidyverse)   # Workhorse with dplyr, ggplot2, etc
library(h2o)         # High Performance Machine Learning
library(recipes)     # Preprocessing
library(rsample)     # Sampling
library(lime)        # Black-box explanations


# 2.0 DATA ----

unzip("00_data/application_train.csv.zip", exdir = "00_data/")
unzip("00_data/HomeCredit_columns_description.csv.zip", exdir = "00_data/")

# Loan Applications (50% of data)
application_train_raw_tbl <- read_csv("00_data/application_train.csv")

application_train_raw_tbl

glimpse(application_train_raw_tbl)


# Column (Feature) Descriptions
feature_description_tbl <- read_csv("00_data/HomeCredit_columns_description.csv")

feature_description_tbl

feature_description_tbl

# 3.0 SIZE THE PROBLEM ----

# How many defaulters?
application_train_raw_tbl %>%
    count(TARGET) %>%
    mutate(n_total = n / 0.15) %>%
    mutate(pct = n_total / sum(n_total)) %>%
    mutate(pct_text = scales::percent(pct))

# Size the problem financially $$$
size_problem_tbl <- application_train_raw_tbl %>%
    count(TARGET) %>%
    filter(TARGET == 1) %>%
    # approximate number of annual defaults
    mutate(prop = 0.15,
           n_total = n / prop) %>%
    # cost of default
    mutate(avg_loan = 15000,
           avg_recovery = 0.40 * avg_loan,
           avg_loss = avg_loan - avg_recovery) %>%
    mutate(total_loss = n_total * avg_loss) %>%
    mutate(total_loss_text = scales::dollar(total_loss))

size_problem_tbl


# 4.0 EXPLORATORY DATA ANALYSIS (SKIPPED) ----
#   SKIPPED - Very Important!
#   Efficient exploration of features to find which to focus on
#   Critical Step in Business Science Problem Framework
#   Taught in my DS4B 201-R Course


# 5.0 SPLIT DATA ----

# Resource: https://tidymodels.github.io/rsample/

set.seed(1234)
split_obj_1 <- initial_split(application_train_raw_tbl, strata = "TARGET", prop = 0.2)
split_obj_2 <- initial_split(training(split_obj_1), strata = "TARGET", prop = 0.8)

# Working with 20% sample of "Big Data"
train_raw_tbl <- training(split_obj_2) # 80% of Data
test_raw_tbl  <- testing(split_obj_2)  # 20% of Data

# Verify proportions have been maintained
train_raw_tbl %>%
    count(TARGET) %>%
    mutate(prop = n / sum(n))

test_raw_tbl %>%
    count(TARGET) %>%
    mutate(prop = n / sum(n))



# 6.0 PREPROCESSING ----

# Fix issues with data: 
#   Some Numeric data with low number of unique values should be Factor (Categorical)
#   All Character data should be Factor (Categorical)
#   NA's (imputation)

# 5.1 Handle Categorical ----

# Numeric
num2factor_names <- train_raw_tbl %>%
    select_if(is.numeric) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value) %>%
    filter(value <= 6) %>%
    pull(key)

num2factor_names

# Character
string2factor_names <- train_raw_tbl %>%
    select_if(is.character) %>%
    names()

string2factor_names


# 6.2 Missing Data ----

# Transform
missing_tbl <- train_raw_tbl %>%
    summarize_all(.funs = funs(sum(is.na(.)) / length(.))) %>%
    gather() %>%
    arrange(desc(value))

missing_tbl

# Visualize
missing_tbl %>%
    filter(value > 0) %>%
    mutate(key = as_factor(key) %>% fct_rev()) %>%
    ggplot(aes(x = value, y = key)) +
    geom_point() +
    geom_segment(aes(xend = 0, yend = key)) +
    expand_limits(x = c(0, 1)) +
    scale_x_continuous(labels = scales::percent) +
    labs(title = "Percentage Missing") 


# 6.3 Recipes ----

# Resource: https://tidymodels.github.io/recipes/

# recipe
rec_obj <- recipe(TARGET ~ ., data = train_raw_tbl) %>%
    step_num2factor(num2factor_names) %>%
    step_string2factor(string2factor_names) %>%
    step_meanimpute(all_numeric()) %>%
    step_modeimpute(all_nominal()) %>%
    prep(stringsAsFactors = FALSE)

# bake
train_tbl <- bake(rec_obj, train_raw_tbl)
test_tbl  <- bake(rec_obj, test_raw_tbl)



# 7.0 MODELING -----

# 7.1 H2O Setup ----

# H2O Docs: http://docs.h2o.ai

h2o.init()

train_h2o <- as.h2o(train_tbl)
test_h2o  <- as.h2o(test_tbl)

y <- "TARGET"
x <- setdiff(names(train_h2o), y)

# 7.2 H2O Models ----

# 7.2.1 GLM (Elastic Net) ----

start <- Sys.time()
h2o_glm <- h2o.glm(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # GLM
    family = "binomial"
    
)
Sys.time() - start

h2o.performance(h2o_glm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7410846

h2o_glm@allparameters

# 7.2.2 GBM ----

# Resource: https://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/

start <- Sys.time()
h2o_gbm <- h2o.gbm(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # GBM
    ntrees = 50,
    max_depth = 5,
    learn_rate = 0.1
)
Sys.time() - start

h2o.performance(h2o_gbm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7415395

h2o_gbm@allparameters

# 7.2.3 Random Forest ----

start <- Sys.time()
h2o_rf <- h2o.randomForest(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # RF
    ntrees = 50,
    max_depth = 5,
    balance_classes = TRUE
    
)
Sys.time() - start

h2o.performance(h2o_rf, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7201136

h2o_rf@allparameters


# 7.2.4 Automated Machine Learning (SKIPPING) ----

#   Deep Learning
#   Stacked Ensembles
#   Grid Search


# 7.3 Saving & Loading Models ----

h2o.saveModel(h2o_automl_se, "00_models")

h2o.loadModel("00_models/StackedEnsemble_AllModels_0_AutoML_20180904_113915")

# 8.0 Making Predictions -----

prediction_h2o <- h2o.predict(h2o_automl_se, newdata = test_h2o)

prediction_tbl <- prediction_h2o %>%
    as.tibble() %>%
    bind_cols(
        test_tbl %>% select(TARGET)
    )


# 9.0 PERFORMANCE (SKIPPING) -----

#   Very Important
#   Adjusting Threshold
#   ROC Plot, Precision vs Recall
#   Gain & Lift - Important for executives


# 10.0 EXPLANATIONS LIME ----

# Create explainer
explainer <- train_tbl %>%
    select(-TARGET) %>%
    lime(
        model           = h2o_gbm,
        bin_continuous  = TRUE,
        n_bins          = 4,
        quantile_bins   = TRUE
    )

# Create explanation
explanation <- test_tbl %>%
    filter(TARGET == "1") %>%
    slice(1) %>%
    select(-TARGET) %>%
    lime::explain(
        explainer = explainer,
        n_features = 8,
        n_permutations = 10000,
        dist_fun = "gower",
        kernel_width   = 1.5,
        feature_select = "lasso_path",
        # n_labels   = 2,
        labels         = "p1"
    )

explanation %>%
    as.tibble() %>%
    glimpse()

# Visualize
plot_features(explanation)


# What are Ext_Source?

feature_description_tbl %>%
    filter(str_detect(Row, "EXT_SOURCE")) %>%
    View()

# Equifax, Experian, TransUnion

# 11.0 OPTIMIZATION (SKIPPING) ----

#   Expected Value
#   Threshold Optimization - Find the balance of False Positives & False Negatives that maximizes revenue
#   Sensitivity Analysis - Taking into account what assumptions we are inputing into the model


# 12.0 RECOMMENDATION ALGORITHMS (SKIPPING) ----

#   3 Step Process:
#       1. Discretized Correlation Visualization (Correlation Funnel)
#       2. Fill out our Recommendation Algorithm Worksheet
#       3. Implement Strategies into R Code
#   Correlation Funnel - S&P Loved This!!






