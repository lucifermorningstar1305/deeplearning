d<-data.frane(x=rnorm(10),y=rnorm(10))
sub_sample <- function(n,d){
  rows <- sample(nrow(d), n)
  d[d<0] <- NA
  d[rows, ]
  
}
d%>%sub_sample(n=3,.)%T>%data.frame(x=.,y=.)%$%data.frame(x=mean(x,na.rm = TRUE), y=mean(y,na.rm=TRUE))



