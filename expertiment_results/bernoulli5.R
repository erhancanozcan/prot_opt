library(data.table)

setwd("/Users/can/desktop/se724_project/simulation")

#Ranking CG Prototype----
data<-read.csv("df_bernoulli5_part1.csv")
data<-as.data.table(data)

data_tmp<-read.csv("df_bernoulli5_part2.csv")
data_tmp<-as.data.table(data_tmp)

data<-rbind(data,data_tmp)
rm(data_tmp)


#select the best algorithm for all seeds.
seed_best<-data[,.(min_best_obj=min(best_obj)),by=.(seed)]
library(dplyr)
data_best_observed_obj_in_seeds<-left_join(data, seed_best, by = c("seed"="seed"))
data_best_observed_obj_in_seeds<-as.data.table(data_best_observed_obj_in_seeds)
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[data_best_observed_obj_in_seeds[,best_obj==min_best_obj],]
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[seed!=6]
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[seed!=11]

data_best_observed_obj_in_seeds_count<-data_best_observed_obj_in_seeds[,.(.N),by=.(Algorithm)]

data<-data[naive_init=="True"]

#select the best algorithm for all seeds for naive_init.
seed_best<-data[,.(min_best_obj=min(best_obj)),by=.(seed)]
library(dplyr)
data_best_observed_obj_in_seeds<-left_join(data, seed_best, by = c("seed"="seed"))
data_best_observed_obj_in_seeds<-as.data.table(data_best_observed_obj_in_seeds)
data_best_observed_obj_in_seeds<-data_best_observed_obj_in_seeds[data_best_observed_obj_in_seeds[,best_obj==min_best_obj],]

data_best_observed_obj_in_seeds_count<-data_best_observed_obj_in_seeds[,.(.N),by=.(Algorithm)]

