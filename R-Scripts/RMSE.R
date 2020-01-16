d <- data.frame(t=rnorm(10), y=rnorm(10))
#(d[,"t"]-d[,"y"])**2
rmse <- d %$% data.frame(t=t, y=y,se=(t-y)**2)%$% mean(.[,'se'])%>%sqrt(.)
rmse