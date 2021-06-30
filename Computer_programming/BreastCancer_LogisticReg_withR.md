# Wisconsin Breast Cancer Data Logistic Regression Analysis

## 1. Load Data


```R
set.seed(2021)

install.packages("mlbench",repos = "http://cran.us.r-project.org")
library(mlbench)
data(BreastCancer)
```

    package 'mlbench' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\CHOI\AppData\Local\Temp\RtmpIBeVmZ\downloaded_packages
    

    Warning message:
    "package 'mlbench' was built under R version 3.6.3"

## 2. EDA


```R
# class of each column
str(BreastCancer)
```

    'data.frame':	699 obs. of  11 variables:
     $ Id             : chr  "1000025" "1002945" "1015425" "1016277" ...
     $ Cl.thickness   : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 5 5 3 6 4 8 1 2 2 4 ...
     $ Cell.size      : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 4 1 8 1 10 1 1 1 2 ...
     $ Cell.shape     : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 4 1 8 1 10 1 2 1 1 ...
     $ Marg.adhesion  : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 1 5 1 1 3 8 1 1 1 1 ...
     $ Epith.c.size   : Ord.factor w/ 10 levels "1"<"2"<"3"<"4"<..: 2 7 2 3 2 7 2 2 2 2 ...
     $ Bare.nuclei    : Factor w/ 10 levels "1","2","3","4",..: 1 10 2 4 1 10 10 1 1 1 ...
     $ Bl.cromatin    : Factor w/ 10 levels "1","2","3","4",..: 3 3 3 3 3 9 3 3 1 2 ...
     $ Normal.nucleoli: Factor w/ 10 levels "1","2","3","4",..: 1 2 1 7 1 7 1 1 1 1 ...
     $ Mitoses        : Factor w/ 9 levels "1","2","3","4",..: 1 1 1 1 1 1 1 1 5 1 ...
     $ Class          : Factor w/ 2 levels "benign","malignant": 1 1 1 1 1 2 1 1 1 1 ...
    


```R
# change all columns' data type into numeric (except "Class" column)
for(i in 1:10){
  BreastCancer[,i]=as.numeric(BreastCancer[,i])
}
```


```R
# change the factor of "Class" [1,2] into [0,1]
#0 for benign, 1 for malignant
BreastCancer$Class=ifelse(BreastCancer$Class=="benign",0,1)
```


```R
# confirm whether the class of each column changed correctly
str(BreastCancer)
```

    'data.frame':	699 obs. of  11 variables:
     $ Id             : num  1000025 1002945 1015425 1016277 1017023 ...
     $ Cl.thickness   : num  5 5 3 6 4 8 1 2 2 4 ...
     $ Cell.size      : num  1 4 1 8 1 10 1 1 1 2 ...
     $ Cell.shape     : num  1 4 1 8 1 10 1 2 1 1 ...
     $ Marg.adhesion  : num  1 5 1 1 3 8 1 1 1 1 ...
     $ Epith.c.size   : num  2 7 2 3 2 7 2 2 2 2 ...
     $ Bare.nuclei    : num  1 10 2 4 1 10 10 1 1 1 ...
     $ Bl.cromatin    : num  3 3 3 3 3 9 3 3 1 2 ...
     $ Normal.nucleoli: num  1 2 1 7 1 7 1 1 1 1 ...
     $ Mitoses        : num  1 1 1 1 1 1 1 1 5 1 ...
     $ Class          : num  0 0 0 0 0 1 0 0 0 0 ...
    


```R
# the number of missing data in each column
for(i in 1:11){
  NaTotal=sum(is.na(BreastCancer[,i]))
  cat(paste(i,":",NaTotal,"\n"))
}
```

    1 : 0 
    2 : 0 
    3 : 0 
    4 : 0 
    5 : 0 
    6 : 0 
    7 : 16 
    8 : 0 
    9 : 0 
    10 : 0 
    11 : 0 
    


```R
# Dealing with Missing data

## Change to median
BreastCancer[is.na(BreastCancer[,7]),7]=median(BreastCancer[,7],na.rm=T)
#BreastCancer$Bare.nuclei[is.na(BreastCancer$Bare.nuclei)]=median(BreastCancer$Bare.nuclei, na.rm=T)

## Change to mean value
#BreastCancer[is.na(BreastCancer[,7]),7]=mean(BreastCancer[,7],na.rm=T)
```

### 3. Logistic Regression Function


```R
myglm=function(Data,lr,epoch,no.variable,no.y){ #no.variable is the vector of the indexes used in regressionriable ex. c(1,5,8)
    # split TestData, Train Data
    n=dim(Data)[1] # total observation of entered data
    k=round(n*0.8)
    l=length(no.variable) # length of variable

    ## xtrain=Data[1:k,no.variable]
    ## xtest=Data[(k+1):n,no.variable]

    ## ytrain=Data[1:k,dim(Data)[2]] #dim(Data)[2] is the number of last variable
    ## ytest=Data[(k+1):n,dim(Data)[2]]

    data.x = Data[,no.variable]
    data.y = Data[,no.y,drop=FALSE]

    train.idx <- sample(1:n, n * 0.80)
    test.idx <- setdiff(1:n, train.idx)

    xtrain <- data.x[train.idx,]
    xtest <- data.x[test.idx,]

    ytrain <- data.y[train.idx,]
    ytest <- data.y[test.idx,]
    
    # matrix for calculation
    Xtrain=as.matrix(xtrain)
    Ytrain=as.matrix(ytrain)
    Xtest=as.matrix(xtest)
    Ytest=as.matrix(ytest)
    #### Gradient Descent
    # deetermining Learning rate, epoch

    # first matrix
    weights=matrix(rep(0,l+1),nrow=l+1) # weight matrix
    Xtrain=cbind(1,Xtrain)
    Xtest=cbind(1,Xtest)

    # logistic (sigmoid) function
    sigmoid=function(z){
      return(1/(1+exp(-z)))
    }
    # Cost Function
    cost=function(x,y,weight){
      z=x%*%weight
      1/dim(x)[1]*(-t(y)%*%log(sigmoid(z))-t((1-y))%*%log(1-sigmoid(z)))
    }
    #### Algorithm training
    # space to save cost and weight
    cost_space=c()
    weight_value=list()

    for(i in 1:epoch){
  
      # delta function
      delta=t(Xtrain)%*%(sigmoid(Xtrain%*%weights)-Ytrain)
  
      # change weight
      weights=weights-lr/k*delta
      cost_space[i]=cost(Xtrain,Ytrain,weights) # save cost
      weight_value[[i]]=weights # save weight
  
      # print log in each 500 times
      if(i%%500==0){
        hypothesis=sigmoid(Xtrain%*%weights)
        prediction=hypothesis>=0.5
    
        correct_prediction=sum(prediction==Ytrain)
        accuracy=correct_prediction/k
    
        cat(paste("Epoch", i, "/", epoch, "Cost:", cost_space[i], "Accuracy:",accuracy, "\n"))
      }
    }
    #### Test accuracy
    test_hypothesis=sigmoid(Xtest%*%weights)
    test_prediction=test_hypothesis>=0.5

    correct_test_prediction=sum(test_prediction==Ytest)
    test_accuracy=correct_test_prediction/dim(Ytest)[1]

    cat("Test Accuracy:",test_accuracy)

    cat("\n \n","< Coefficient >", "\n")
    print(weight_value[[epoch]])

    weight.result=as.matrix(weight_value[[epoch]])

    return(weight.result)

}
```

### 4. Check with Breat Cancer Data


```R
BCweight=myglm(BreastCancer,0.01,5000,2:10,11)
```

    Epoch 500 / 5000 Cost: 0.319551382832637 Accuracy: 0.92128801431127 
    Epoch 1000 / 5000 Cost: 0.269504725860659 Accuracy: 0.930232558139535 
    Epoch 1500 / 5000 Cost: 0.23982973960425 Accuracy: 0.9391771019678 
    Epoch 2000 / 5000 Cost: 0.218227486200613 Accuracy: 0.942754919499106 
    Epoch 2500 / 5000 Cost: 0.201373651112068 Accuracy: 0.948121645796064 
    Epoch 3000 / 5000 Cost: 0.187779369427817 Accuracy: 0.953488372093023 
    Epoch 3500 / 5000 Cost: 0.176583076789662 Accuracy: 0.955277280858676 
    Epoch 4000 / 5000 Cost: 0.167218197001144 Accuracy: 0.955277280858676 
    Epoch 4500 / 5000 Cost: 0.159286064023044 Accuracy: 0.953488372093023 
    Epoch 5000 / 5000 Cost: 0.152494762948518 Accuracy: 0.955277280858676 
    Test Accuracy: 0.9642857
     
     < Coefficient > 
                           [,1]
                    -3.26868163
    Cl.thickness     0.03199420
    Cell.size        0.44296137
    Cell.shape       0.13139005
    Marg.adhesion    0.10572249
    Epith.c.size    -0.28560172
    Bare.nuclei      0.44872860
    Bl.cromatin     -0.16119126
    Normal.nucleoli  0.20753372
    Mitoses          0.01143986
    

### 5. Predict with New Data


```R
mypredict=function(newX,weight){
  newX=c(1,newX)
  newX=t(as.matrix(newX))
  result=ifelse(newX%*%weight>=0,"malignant","benign")
  cat("The patient is ", result)
    
}
```


```R
mypredict(c(9,8,9,8,9,8,9,8,1),BCweight)
```

    The patient is  malignant
