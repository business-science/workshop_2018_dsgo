# CHALLENGE SOLUTION ----

start <- Sys.time()
h2o_deeplearning <- h2o.deeplearning(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # Deep Learning
    epochs = 10,
    hidden = c(100, 50, 10)
)
Sys.time() - start
# Time difference of 59.41523 secs

h2o_deeplearning %>% h2o.auc(valid = TRUE)
# [1] 0.7098785
