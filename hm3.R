###############################
## Association Rule Mining
## Bank Data
## Personal Equity Plan (PEP)
## hmw3
## TJ
##############################


## load library
library(plyr)
library(dplyr)
library(arules)
library(arulesViz)

## load bank datasets
bankdata <- read.csv('../data/wk3-data/bankdata_csv_all.csv', na.strings = (''))

## print top ten
head(bankdata,10)

## summary
summary(bankdata)

## structure of the data
str(bankdata)

## Converting record data to transaction data
## he first step is to convert all numeric variables to nominal
## the bank data might have duplicate items like "NO, NO, NO, NO",
## which should be converted to "married=NO, car=NO, save_act=NO, current_act=NO".


# Data Transformation
## discretized the age
bankdata$age <- cut(bankdata$age, breaks = c(0,12,30,50,70,Inf),labels=c("child","young_age","middle_age","senior_age","old"))

#check
table(bankdata$age)

## discretize the income bin
# Using equal bin
summary(bankdata$income)

max_income <- max(bankdata$income)
min_income <- min(bankdata$income)
bins <- 3
width <- (max_income - min_income)/bins
bankdata$income <- cut(bankdata$income, breaks = seq(min_income,max_income, width))
# check
table(bankdata$income)


## convert numeric to nominal for the children columns
bankdata$children <- as.factor(bankdata$children)
# check
table(bankdata$children)

##
## Now the second step of conversion, changing "YES" to "[variable_name]=YES".
bankdata$married <- dplyr::recode(bankdata$married, YES='married=YES', NO='married=NO')
bankdata$car <- dplyr::recode(bankdata$car, YES='car=YES',NO='car=No')
bankdata$save_act <- dplyr::recode(bankdata$save_act, YES='save_act=YES', NO='save_act=NO')
bankdata$current_act <- dplyr::recode(bankdata$current_act, YES='current_act=YES', NO='current_act=NO')
bankdata$mortgage <-dplyr::recode(bankdata$mortgage, YES='mortgage=YES', NO='mortgage=NO')
bankdata$pep <- dplyr::recode(bankdata$pep, YES='pep=YES', NO='pep=NO')

# print head
head(bankdata)

## structure
str(bankdata)
## 
Tester <- bankdata[,-1] #remove the id column
head(Tester)


## Create rule
rules <- apriori(Tester,parameter = list(supp=0.001, conf=0.8, maxlen=3))
inspect(rules[1:5])

# sort the rule with confidence
rules <- sort(rules, by='confidence', decreasing = TRUE)
inspect(rules[1:5])


#create a rule targeting pep=NO
rules <- apriori(Tester, parameter = list(supp=0.001, conf=0.8, minlen=2),
                 appearance = list(default='lhs', rhs='pep=pep=NO'),
                 control = list(verbose=FALSE))

#sort and inspect rules
rules <- sort(rules, by='confidence', decreasing = TRUE)
inspect(rules[1:5])

# target pep=yes
rules <- apriori(Tester, parameter = list(supp=0.001, conf=0.8, minlen=2),
                 appearance = list(default='lhs', rhs='pep=pep=YES'),
                 control = list(verbose=FALSE))

summary(rules)  # get the summary details of the rules


#sort and inspect rules
rules <- sort(rules, by='confidence', decreasing = TRUE)
inspect(rules[1:5])

#sort rule by lift
rules <- sort(rules, by='lift', decreasing = TRUE)
inspect(rules[1:5])

#sort rule bysupport
rules <- sort(rules, by='support', decreasing = TRUE)
inspect(rules[1:5])


#visualiz
# top 5 rules
subrules <- head(rules,5)
plot(subrules,method = 'graph', interactive = TRUE)

##viz
subrules <- head(rules,5)
plot(subrules,method = 'graph')


