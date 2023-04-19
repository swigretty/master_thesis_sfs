library(sandwich)
library(lmtest)
library(zoo)
library(nlme)
library(insight)

simulate <- function(ndays = 4, samples_per_hour = 10, phase = pi, 
                     amplitude = 2, b0 = 100, b1 = 0, ar = c(0.8)){
  period = 24 * samples_per_hour
  n_samples = (period * ndays)
  t <- 1:n_samples
  freq = 1/period
  
  b2 = amplitude * cos(phase)
  b3 = amplitude * sin(phase)
  b = c(b0, b1, b2, b3)
  X = cbind(1, t, cos(2*pi*t*freq), sin(2*pi*t*freq))
  Xdf = data.frame(X)
  colnames(Xdf) <- c("(Intercept)", "t", "cos", "sin")
  
  start <- as.POSIXct("2023-01-01 00:00:00", tz="utc")
  end <- start + as.difftime(ndays, units="days")
  dts <- seq(from=start, length.out=n_samples, to=end)
  
  ar <- arima.sim(list(ar=ar), n_samples)
  
  
  return(list(dts=dts, X=Xdf, b=b, ar=ar))
}


remove_data_points <- function(n_samples, kept_data_fraction=0.5, prob=NULL)
{
  t.red <- sort(sample(1:n_samples, round(n_samples*kept_data_fraction), 
                       replace = FALSE, 
                       prob=prob))
  return(t.red)
}


plot_predictions <- function(dts_plot, time, y, t.hat, y.hat, y.hat.lw, y.hat.up, main=""){
  
  day_hour <- paste(format(dts_plot, "%d"), format(dts_plot, "%H"), sep=":")
  xaxis_at <- seq(1, length(dts_plot), round(length(dts_plot)/10))
  
  plot(time, y, type="l", lwd=0.1, main=main, xaxt="n", xlim = range(1:length(dts_plot)))
  lines(t.hat, y.hat, col="red",lty=1)
  lines(t.hat, y.hat.lw, col="blue",lty=2)
  lines(t.hat, y.hat.up, col="blue",lty=2)
  axis(1, at=xaxis_at, labels=day_hour[xaxis_at],  las=2)
  
}


plot_lm_prediction <- function(fm, dts_plot, Xfitted, vcov_estimation=NULL, X=NULL, 
                               y=NULL, main="", predict="expectation") {
  time <- Xfitted[, "t"]
  if (is.null(X)) {X <- Xfitted}
  t.hat <- X[, "t"]
  if (is.null(y)) {y <- fm$model$y}
  ci <- as.data.frame(get_predicted(fm, data=X, ci=0.95, predict=predict, vcov_estimation=vcov_estimation))
  plot_predictions(dts_plot, time, y, t.hat, ci[, "Predicted"], ci[, "CI_low"], ci[,"CI_high"], main=main)
  return(ci)
  
  # 
  # Xfitted <- model.matrix(fm)
  # 
  # if (is.null(y)) {y <- fm$model$y}
  # 
  # if (is.null(X)) {X <- Xfitted}
  # 
  # if (is.null(vcov)) {vcov <- vcov(fm)}
  # 
  # p <- length(fm$coefficients)
  # n <- length(y)
  # t <- Xfitted[, "t"]
  # t.hat <- X[, "t"]
  # 
  # y.hat <- X %*% fm$coefficients
  # sigma.hat.sq = sum(fm$residuals^2)/(n-p)
  # 
  # se.y.hat = sqrt(diag(X%*%vcov%*%t(X)))
  # # for ols: se.y.hat <- sigma.hat * sqrt(diag(X %*% solve(t(X) %*% X) %*% t(X)))
  # if (ci_type == "prediction") {se.y.hat <- sqrt(sigma.hat.sq + diag(X%*%vcov%*%t(X)))}
  # 
  # y.hat.lw <- y.hat - qt(0.975, (n-p)) * se.y.hat
  # y.hat.up <- y.hat + qt(0.975, (n-p)) * se.y.hat
  # 
  # plot_predictions(dts_plot, t, y, t.hat, y.hat, y.hat.lw, y.hat.up, main=main)
  # 
  # return(data.frame(Predicted=y.hat, CI_low=y.hat.lw, CI_hihg=y.hat.up))
}



ar = c(0.8)
amplitude = 4

data <- simulate(ar=ar, amplitude=amplitude)
n_samples = length(data$dts)

seasonal_true = as.matrix(data$X[, 3:4]) %*% data$b[3:4]
plot(data$dts, seasonal_true, type='l')
plot(data$dts, data$ar, type="l")

y <- as.matrix(data$X) %*% data$b + data$ar + rnorm(n_samples, sd=1)
plot(data$dts, y, type="l")

fit.lm.full <- lm(y ~ t + cos + sin, data=data$X)


# alternative
# library(mgcv)
# fit.gam.full <- gam(y ~ s(t) + cos + sin, data=data$X)


# library(mgcv)
# tr <- as.numeric(time(co2))
# months <- rep(c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
#                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
# months <- factor(rep(months,40),levels=months)[1:length(co2)]
# fit <- gam(co2 ~ s(tr) + months)
# plot(co2, ylab="co2",type="l")
# lines(tr, fitted(fit), col="red")


corStruct <- corARMA(form=~t, p=length(ar))
fit.gls <- gls(y ~ t + cos + sin, data=data$X, corr=corStruct)
# fit.gls
# acf(resid(fit.gls), main="ACF of GLS-Residuals")
# pacf(resid(fit.gls), main="PACF of GLS-Residuals")

kept_data_prob = rep(1, n_samples) + 1000000 * (seasonal_true + amplitude)
kept_data_prob = kept_data_prob/sum(kept_data_prob)
t.red <- remove_data_points(n_samples, prob=kept_data_prob)

y.red <-y[t.red]
data.red <- list(dts=data$dts[t.red], X=data$X[t.red,])
par(mfrow=c(2,1))

plot(data.red$X[,"t"], y.red)
hist(data.red$X[,"t"], breaks=50)

fit.lm.red <- lm(y.red ~ t + cos + sin, data=data.red$X)

corStruct <- corARMA(form=~t, p=length(ar))
fit.gls.red <- gls(y.red ~ t + cos + sin, data=data.red$X, corr=corStruct)





# plot(fm)
plot(data.red$dts, fit.lm.red$residuals, type="l")
acf(fit.lm.red$residuals)
pacf(fit.lm.red$residuals)

# , "prediction"

for (predict in c("expectation")){
  
  svg(paste(predict, "svg", sep = "."))
  
  par(mfrow=c(2,3))
  
  
  
  for (mode in list(list(fm=fit.lm.full, main="lm.full"), list(fm=fit.lm.full, vcov_estimation="NeweyWest",  main="lm.full.nw"),
                    list(fm=fit.gls, main="gls.full")
  )){
    mode$Xfitted <- data$X
    mode$dts_plot <- data$dts
    mode$y <- y
    mode$predict <- predict
    ci <- do.call(plot_lm_prediction, mode)
  }
  
  for (mode in list(list(fm=fit.lm.red, main="lm.red"), list(fm=fit.lm.red, vcov_estimation="NeweyWest",  main="lm.red.nw"),
                    list(fm=fit.gls.red, main="gls.red")
  )){
    mode$Xfitted <- data.red$X
    mode$dts_plot <- data$dts
    mode$X <- data$X
    mode$y <- y.red
    mode$predict <- predict
  
    ci <- do.call(plot_lm_prediction, mode)
  }
  dev.off()
  
}


# 
# par(mfrow=c(2,3))
# 
# xaxis_at <- seq(1, length(y), round(length(y)/10))
# 
# plot(data$X[,"t"], y, type="l", lwd=0.1, xaxt="n")
# axis(1, at=xaxis_at, labels=day_hour[xaxis_at],  las=2)
# 
# 
# plot(data$dts, (ci.lm.full[,2]- ci.lm.full[,1]),type="l", main="lm.full.hac")
# 
# # plot(data$dts, ci[,3]- ci[,2])
# plot(dts.red, y.red, type="l", lwd=0.1)
# hist(dts.red, breaks=50)
# 
# plot(data$dts, ci.lm.full.hac[,2]- ci.lm.full.hac[,1])
# 
# plot(data$dts, ci.gls.full[,2]- ci.gls.full[,1])
# 
# plot(data$dts, ci.gls.full[,2]- ci.gls.full[,1])
# 
# 
