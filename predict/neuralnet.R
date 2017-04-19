train.df = read.csv('../data/train.csv')
test.df = read.csv('../data/test.csv')
train.df['id'] = NULL
summary(train.df)
dim(train.df)
head(train.df)
sum(is.na(train.df)) # No Missing values
colnames(train.df[, 117:130])

#0.5. Some extra steps from Kaggle
library(data.table)
library(Matrix)

ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200

TRAIN_FILE = "../data/train.csv"
TEST_FILE = "../data/test.csv"

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[, TARGET, with = FALSE] + SHIFT)[[TARGET]]

train[, c(ID) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
# train_test = rbind(train, test)

features = names(train)

for (f in features) {
  if (class(train[[f]]) == "character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  }
}

# In order to speed up fit within Kaggle scripts have removed 30
# least important factors as identified from local run
features_to_drop <- c("cat67","cat21","cat60","cat65", "cat32", "cat30",
                      "cat24", "cat74", "cat85", "cat17", "cat14", "cat18",
                      "cat59", "cat22", "cat63", "cat56", "cat58", "cat55",
                      "cat33", "cat34", "cat46", "cat47", "cat48", "cat68",
                      "cat35", "cat20", "cat69", "cat70", "cat15", "cat62")

train.df = train[1:ntrain, -features_to_drop, with = FALSE]
train.df = as.data.frame(train.df)

for (f in features) {
  if (class(test[[f]]) == "character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(test[[f]]))
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}
test.df = test[1:nrow(test), -features_to_drop, with = FALSE]




# 1. Normalize numerical features
normalize = function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

train.df[, 117:130] = as.data.frame(lapply(train.df[, 117:130], normalize))
summary(train.df[, 117:130])
dim(train.df[, 117:130])
apply(train.df[, 117:130], 2, hist)

# 2. split data into training set (80%) and test set (20%)
set.seed(0)
train.train = train.df[1:150655, ]
train.test = train.df[150655:188318, ]
dim(train.train)


# 3. create formula
formula = as.formula(paste("loss ~", paste(names(train.df[, -length(train.df)]), collapse = " + ")))


# 4. train model using the `neuralnet` function
library(neuralnet)
nn1 = neuralnet(formula, hidden = 10, data = train.df)
# nn2 = neuralnet(formula, hidden = c(10, 10, 5), data = train.train)


# 5.1 make predictions on test set
predict_nn1 = compute(nn1, test.df)
# predict_nn2 = compute(nn2, train.test[, 1:130])


# 5.2 calculate MSE of the test set
library(hydroGOF)
MSE.nn1 = mse(predict_nn1$net.result, as.data.frame(train.test[, 131]))
print(MSE.nn1)
MSE.nn2 = mse(predict_nn2$net.result, as.data.frame(train.test[, 131]))
print(MSE.nn2)