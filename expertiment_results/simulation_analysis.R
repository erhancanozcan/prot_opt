library(data.table)

setwd("/Users/can/desktop/se724_project/simulation")



data<-read.csv("df_bernoulli3.csv")
data<-as.data.table(data)



#select the best algorithm for all seeds.
seed_best<-data[,.(min_best_obj=min(best_obj)),by=.(seed)]
library(dplyr)
data_best_observed_obj_in_seeds<-left_join(data, seed_best, by = c("seed"="seed"))
data_best_observed_obj_in_seeds<-as.data.table(data_best_observed_obj_in_seeds)
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[data_best_observed_obj_in_seeds[,best_obj==min_best_obj],]

data_best_observed_obj_in_seeds_count<-data_best_observed_obj_in_seeds[,.(.N),by=.(Algorithm)]

#analysis on naive initialization.

data<-read.csv("df_bernoulli3.csv")
data<-as.data.table(data)

data<-data[naive_init=="True"]

#select the best algorithm for all seeds for naive_init.
seed_best<-data[,.(min_best_obj=min(best_obj)),by=.(seed)]
library(dplyr)
data_best_observed_obj_in_seeds<-left_join(data, seed_best, by = c("seed"="seed"))
data_best_observed_obj_in_seeds<-as.data.table(data_best_observed_obj_in_seeds)
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[data_best_observed_obj_in_seeds[,best_obj==min_best_obj],]

data_best_observed_obj_in_seeds_count<-data_best_observed_obj_in_seeds[,.(.N),by=.(Algorithm)]











####
data<-read.csv("df_bernoulli1.csv")
data<-as.data.table(data)
#count_num_obs<-data[,.(count_num=.N),by=.(Algorithm,naive_init,momentum,lr)]
better_obj_data<-data[data[,best_obj<max_data_obj],]
count_num_improvement_data<-better_obj_data[,.(count_num_imp_data=.N),by=.(Algorithm,naive_init,momentum,lr)]
count_num_improvement_data<-count_num_improvement_data[naive_init==TRUE]
#In 0.9 setting no algorithm manage to improve max_data_obj with naive initialization.
#However, all algorithms can beat max data obj by selecting an appropriate learning rate
#when max init is employed.


####
data<-read.csv("df_bernoulli3.csv")
data<-as.data.table(data)
#count_num_obs<-data[,.(count_num=.N),by=.(Algorithm,naive_init,momentum,lr)]
better_obj_data<-data[data[,best_obj<max_data_obj],]
count_num_improvement_data<-better_obj_data[,.(count_num_imp_data=.N),by=.(Algorithm,naive_init,momentum,lr)]
count_num_improvement_data<-count_num_improvement_data[naive_init=="True"]
#write.csv(count_num_improvement_data,'ber3.csv')
#In 0.5 setting, when max init is employed, by selecting the optimization algorithms'
#parameters carefully, they can beat max_data_obj. Interesting case, there are some
#situations than can naive initialization can beat max_data_obj


####
data<-read.csv("df_bernoulli5_part1.csv")
data<-as.data.table(data)

data_tmp<-read.csv("df_bernoulli5_part2.csv")
data_tmp<-as.data.table(data_tmp)

data<-rbind(data,data_tmp)
rm(data_tmp)
#count_num_obs<-data[,.(count_num=.N),by=.(Algorithm,naive_init,momentum,lr)]
better_obj_data<-data[data[,best_obj<max_data_obj],]
count_num_improvement_data<-better_obj_data[,.(count_num_imp_data=.N),by=.(Algorithm,naive_init,momentum,lr)]
count_num_improvement_data<-count_num_improvement_data[naive_init=="True"]
#write.csv(count_num_improvement_data,'ber5.csv')

