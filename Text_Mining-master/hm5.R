########################################
# Tajudeen Abdulazeez
# IST 565 Data Mining
# tabdulazeez99@gmail.com
# www.toraaglobal.com
#######################################
## Mystery in History
## Fedpapers
## Authorship betweem Madison and Hamilton
## Aims is to solve the authorship dispute between Madison and Hamilton
## Analysis Aaproached
#         Decision Tree
#         Clustering 
#              Expectation Maximization
#              K-Means
#              HAC : Hirachical Clustering
#              DBSCAN : Density Clustering


## Lload libraries
library(ggplot2)
library(mclust)
library(cluster)
library(arules)

library(unbalanced)
library(RWeka)
library(CORElearn)


library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
#library(Cairo)

library(network)
library(wordcloud)
library(tm)
library(slam)
library(quanteda)
library(SnowballC)
library(proxy)
library(stringr)
library(textmineR)
library(igraph)
library(lsa)


## Load Fedpapers
FedFilesPath <- "../data/FedCorpus_original"

FedCorpus <- Corpus(DirSource(FedFilesPath))
FedCorpus
### the content of the corpus: 85 documents

## file list
FilesList <- list.files(FedFilesPath, pattern=NULL)
FilesList

## Text Pre-processing
# getTransformations() : [1] "removeNumbers"     "removePunctuation" "removeWords"       "stemDocument"      "stripWhitespace"


# Tranform all the text in the document together

cleanFedCorpus <- tm_map(FedCorpus, removePunctuation)  # remove all the punctuation from the documents
cleanFedCorpus <- tm_map(cleanFedCorpus, removeNumbers) # remove all numbers from the documents
cleanFedCorpus <- tm_map (cleanFedCorpus, content_transformer(tolower)) # convert all text to lower case
cleanFedCorpus <- tm_map(cleanFedCorpus, removeWords,stopwords('english')) # remove all stopwords, using quenteda packages

cleanFedCorpus <- tm_map(cleanFedCorpus, stripWhitespace) #remove whitespace
#cleanFedCorpus <- tm_map(cleanFedCorpus, content_transformer(removeURL))

# apply lementization
cleanFedCorpus <- tm_map(cleanFedCorpus,stemDocument)

# inspect the documents
tm::inspect(cleanFedCorpus[1:5])



# create a dataframe
fedDataFrame <- data.frame(text= sapply(cleanFedCorpus, identity), stringsAsFactors = FALSE)

# write the fedDataframe to csv
write.csv(fedDataFrame, '../data/fedCorpusClean.csv')


head(fedDataFrame,1)

#number of rows
cat('The total number of rows : ',nrow(fedDataFrame))

# columns
colnames(fedDataFrame)

length(row.names(fedDataFrame))

## create a new columns called arthur
Tester <- fedDataFrame

# Tester$author <- ifelse(grepl("Hamilton_fed_",row.names(Tester), ignore.case = T), 'Hamilton',
#                         ifelse(grepl('Madison_fed_', row.names(Tester), ignore.case = T),'Madison'),
#                         ifelse(grepl('Jay_fed_', row.names(Tester),ignore.case = T), 'Jay'),
#                         ifelse(grepl('Hm_fed_', row.names(Tester),ignore.case = T),'Hm'),
#                         ifelse(grepl('dispt_fed_', row.names(Tester),ignore.case = T),'Dispt', "Other"))
# 


Tester$author <- ifelse(grepl("Hamilton_fed_",row.names(Tester), ignore.case = T),'Hamilton',
                        ifelse(grepl('Madison_fed_', row.names(Tester), ignore.case = T),'Madison', 'other'))


Tester$author <- ifelse(grepl('Jay_fed_', row.names(Tester),ignore.case = T),'Jay',
                        ifelse(grepl('Hm_fed_', row.names(Tester),ignore.case = T),'Hm', Tester$author))

#dispt
Tester$author <- ifelse(grepl('dispt_', row.names(Tester), ignore.case = T),'Dispt',Tester$author)

#check for unique author
unique(Tester$author)


#check the frequency distribution of authors
table(Tester$author)
# 
# Dispt Hamilton       Hm      Jay  Madison 
# 11       51        3        5       15 

fedDataFrame <- Tester # copy the prepared Tester back to fedDataFrame

str(fedDataFrame) # dataframe with 85 observation and two variables

#####################################################
## Document Matrix
##
####################################################
# Term document matrix
fedTDM <- TermDocumentMatrix(cleanFedCorpus)
tm::inspect(fedTDM)

# find most frequent words
findMostFreqTerms(fedTDM,1)


# convert the matrix to dataframe
fedTDM

# <<TermDocumentMatrix (terms: 4923, documents: 85)>>
#   Non-/sparse entries: 45973/372482
# Sparsity           : 89%
# Maximal term length: 18
# Weighting          : term frequency (tf)

# find associations with a selected conf
findAssocs(fedTDM, 'state', 0.50)

## Dataframe
fedTDMDf <- as.data.frame(tm::inspect(fedTDM))
dim(fedTDMDf)

#scale
fedTDMDfScale <- scale(fedTDMDf)

## HAC   : Clustering
# Distance metrics
#               euclidean
#               manhattan
#               cosine

d <- dist(fedTDMDfScale, method = 'euclidean')
model_HAC <- hclust(d, method = 'ward.D2')
plot(model_HAC)




# Document Term Matrix
fedDTM <-DocumentTermMatrix(cleanFedCorpus)
tm::inspect(fedDTM)
fedDTM

# <<DocumentTermMatrix (documents: 85, terms: 4923)>>
#   Non-/sparse entries: 45973/372482
# Sparsity           : 89%
# Maximal term length: 18
# Weighting          : term frequency (tf)

fedDTMDf <- as.data.frame(tm::inspect(fedDTM))
dim(fedDTMDf)
fedDTMDfScale <- scale(fedDTMDf)

## HAC
d <- as.matrix(fedDTMDfScale, method = 'euclidean')
model_HAC <- hclust(d, method = 'ward.D2')
plot(model_HAC)


####################################################
## Normalizaed and Re-visualized
##
###################################################

normalizedTDM <- TermDocumentMatrix(cleanFedCorpus, control = list(weighting = weightTfIdf, stopwords = TRUE))
tm::inspect(normalizedTDM)

# viz
cleadDf_N <-as.data.frame(tm::inspect(normalizedTDM))

# scale
cleadDf_N_Scale <- scale(cleadDf_N)

d <- dist(cleadDf_N_Scale, method = 'euclidean')
model_HAC <- hclust(d, method = 'ward.D2')

rect.hclust(model_HAC, k=4)  # cut the three to 4 cluster
plot(model_HAC)



##################################################################
##
## WORDCLOUD
##
##################################################################

# normalized term document matrix
m <- as.matrix(normalizedTDM)
#m

# calculate the freq of words
word_freq <- sort(rowSums(m), decreasing = T)

wordcloud(words = names(word_freq), freq = word_freq * 2, min.freq =10, random.order = F )



### DATA PREP FOR MODELING

Tester <- fedDataFrame

head(Tester,1)

table(Tester$author)

## Since the disbute is between madison and Hamilton. HM and Jay is excluded from the subset used in the analysis.

Tester <- subset(Tester, Tester$author== 'Madison' | Tester$author == 'Hamilton' | Tester$author =='Dispt')


table(Tester$author)
dim(Tester)

# Dispt Hamilton  Madison 
# 11       51       15

nrow(Tester)

# Now we  have 77 documents.



## CREATE A CORPUS FROM THE DATAFRAME
newFedCorpus <- Corpus(VectorSource(Tester$text))

newFedCorpus
# 
# <<SimpleCorpus>>
#   Metadata:  corpus specific: 1, document level (indexed): 0
# Content:  documents: 77

tm::inspect(newFedCorpus[1:5])

## Document Matrix

newFedCorpusDTM <- DocumentTermMatrix(newFedCorpus, control = list(weighting = weightTfIdf, stopwords = T)) # normalized
newFedCorpusTDM <- TermDocumentMatrix(newFedCorpus, control = list(weighting = weightTfIdf, stopwords = T)) # normalized
tm::inspect(newFedCorpusDTM)
tm::inspect(newFedCorpusTDM)



## HAC
newCleanDF_N <- as.data.frame(tm::inspect(newFedCorpusDTM))
dim(newCleanDF_N)

newCleanDF_N_Scale <- scale(newCleanDF_N)

head(newCleanDF_N_Scale,5)

d <- dist(t(newCleanDF_N_Scale), method = 'euclidean')  # transpose the dataframe during dist calculation
model_HAC <- hclust(d, method = 'ward.D2')
rect.hclust(model_HAC, k=4) # cut tree to four clusters
plot(model_HAC)


## WordCloud

m <- as.matrix(newFedCorpusTDM)

#calculate word freq
word_freq <- sort(rowSums(m),decreasing = T)

wordcloud(words = names(word_freq), freq = word_freq, min.freq = 10, random.order = F)



################################
####                                Corpus from Datafram
################################


corpusDf <- Corpus(VectorSource(Tester$text))
tm::inspect(corpusDf)

corpusDf

# Document term

corpusDf <- DocumentTermMatrix(corpusDf)

corpusDf <- as.data.frame(tm::inspect(corpusDf))

dim(corpusDf)


###############################################################################################
##
## Read from csv
##
###############################################################################################

fedpapers <- read.csv('../data/wk4-data/fedPapers85.csv')
str(fedpapers)
summary(fedpapers)

colnames(fedpapers)

#freq
table(fedpapers$author)

summary(fedpapers)

## 
row.names(fedpapers)

###  Remove label
fedpapers_unlabeled <- fedpapers[,-c(1,2)]
str(fedpapers_unlabeled)

#plot of authour and document count
p <- ggplot(fedpapers,aes(author,filename) ) + geom_count(color='red', fill='blue', size=7) + theme_classic()
p


summary(fedpapers_unlabeled)


### WordCloud #####
names(fedpapers_unlabeled)

wordcloud(words = names(fedpapers_unlabeled), freq = col_sums(fedpapers_unlabeled) * 100, random.order = F)

#
head(fedpapers,1)

table(fedpapers$author)

###############################################
set.seed(123)

Tester <- fedpapers

Tester <- Tester[,-2]

typeof(Tester)

# for supervesied learning, create a test and training set
test <- subset(Tester, Tester$author == 'dispt')  # only the disputed article for test
unique(test$author)
test$author
colnames(test)
nrow(test) 

#train
train <- subset(Tester, Tester$author == 'Madison' | Tester$author == 'Hamilton')

nrow(train)


#
pairs(train[,c(1:5)])



# 
summary(train)

summary(test)

dim(train)

typeof(train)
###################################################### tree ##############
# create a decision tree model

model <- rpart(author ~ ., data=train, method = 'class')

summary(model)
        
model <- rpart(author ~ ., data=train, method = 'class', control = rpart.control(minsplit = 2,cp=0.01))

summary(model)

fancyRpartPlot(model)
#*************************information gain ********************************
InfoGainAttributeEval(author ~., data=train)

##################################CORE learn

model1 <- CORElearn::attrEval(train$author ~ .,data=train, estimator = 'InfGain')
model

model2 <- CORElearn::attrEval(train$author ~ .,data=train, estimator = 'Gini')
model2

model3 <- CORElearn::attrEval(train$author ~ .,data=train, estimator = 'GainRatio')
model3



###########################################
model <- rpart(author ~ upon + there +on + by + to + and, data=train, method = 'class', control = rpart.control(minsplit = 2,cp=0.01))
summary(model)
fancyRpartPlot(model)



#######################predict 
head(test)

# remove author
test <- test[,-1] # remove author from test set

#make a prediction
predicted <- predict(model, test, type='class')
predicted

##################################
df <- subset(fedpapers, fedpapers$author == 'dispt', select = c(filename,author))
df

############## make prediction
df$predicted <- predicted
df
