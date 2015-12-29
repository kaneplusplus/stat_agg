suppressMessages(library(randomForest))
suppressMessages(library(foreach))
suppressMessages(library(MASS))
set.seed(1)

# Get the airline delay data
airline_file = tempfile()

if (!file.exists(airline_file)) {
  download.file("http://stat-computing.org/dataexpo/2009/2008.csv.bz2",
                destfile=airline_file)
}

x = read.csv(bzfile(airline_file))
#x = x[sample.int(nrow(x), nrow(x)),]
x = x[x$DepDelay >= 0,]

train_inds = sample(1:nrow(x), round(0.5*nrow(x)))
test_inds = setdiff(1:nrow(x), train_inds)

x$Late = as.numeric(x$DepDelay > 15)

# Write out the test data
write.csv(x[test_inds,], "test_airline.csv")

train_data = x[train_inds,]
test_data = x[test_inds,]

regressor_names = c("Month", "DayofMonth", "DayOfWeek", "CRSDepTime", 
                    "ArrTime", "CRSArrTime", "UniqueCarrier", 
                    "ActualElapsedTime", "CRSElapsedTime", "Distance")

# 10 Linear regressors, 5 variables a piece.

lm_fits = foreach(i=1:10) %do% {
  regs = sample(regressor_names, 5)
  form = as.formula( paste("DepDelay", paste(regs, collapse=" + "), sep=" ~ "))
  lm(form, train_data)
}

fitted_values = lapply(lm_fits, function(x) predict(x, train_data))

fitted_df = as.data.frame(Reduce(cbind, fitted_values))
names(fitted_df) = paste0("learner", 1:ncol(fitted_df))
fitted_df$actual = train_data$DepDelay
fitted_df = na.omit(fitted_df)
write.csv(fitted_df, "regression_predictions_train.csv", row.names=FALSE)

# Test data
fitted_df = as.data.frame(lapply(lm_fits, function(x) predict(x, test_data)))

names(fitted_df) = paste0("learner", 1:ncol(fitted_df))
fitted_df$actual = test_data$DepDelay
fitted_df = na.omit(fitted_df)
write.csv(fitted_df, "regression_predictions_test.csv", row.names=FALSE)

logit_fits = foreach(1:10) %do% {
  regs = sample(regressor_names, 5)
  form = as.formula( paste("Late", paste(regs, collapse=" + "), sep=" ~ "))
  glm(form, train_data, family=binomial)
}

fitted_values = lapply(logit_fits, 
                       function(x) round(predict(x,train_data,type="response")))
fitted_df = as.data.frame(Reduce(cbind, fitted_values))
names(fitted_df) = paste0("learner", 1:ncol(fitted_df))
fitted_df$actual = train_data$Late
write.csv(na.omit(fitted_df), "classification_predictions_train.csv", 
          row.names=FALSE)

fitted_values = lapply(logit_fits, 
                       function(x) round(predict(x,test_data,type="response")))
fitted_df = as.data.frame(Reduce(cbind, fitted_values))
names(fitted_df) = paste0("learner", 1:ncol(fitted_df))
fitted_df$actual = test_data$Late
write.csv(na.omit(fitted_df), "classification_predictions_test.csv", 
          row.names=FALSE)

