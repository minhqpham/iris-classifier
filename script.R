irisdata <- read.csv(file = 'irisdata.csv')
plotdata <- irisdata %>% filter(species == "versicolor" | species == "virginica")
plot1 <- ggplot(plotdata, aes(petal_length, petal_width, color = species)) + geom_point() + labs(x = "Petal length", y = "Petal width", title = "Petal Width vs Petal Length, Versicolor and Virginica", color = "Species") 

compute_logistics <- function(w0, w1, w2, petal_length, petal_width) {
  y <- w0 + w1*petal_length + w2*petal_width
  z <- 1/(1 + exp(-y))
  return (z)
}

classifier <- function(w0, w1, w2, x1, x2) {
  x <- compute_logistics(w0, w1, w2, x1, x2) 
  if (x < 0.5)
    return (0)
  else
    return(1)
}

boundary <- function(w0, w1, w2, x1) {
  x2 <- (w0 + w1*x1)/(-w2)
  return (x2)
}

plot2 <- plot1 + stat_function(fun = boundary, args = list(w0 = -2.74, w1 = 0.24, w2 = 1))

w0_f = -2.74
w1_f = 0.24
w2_f = 1

x <- seq(0, 10, by = 0.1)
y <- seq(0, 10, by = 0.1)
z <- matrix(data = NA, nrow = length(x), ncol = length(y), byrow = FALSE, dimnames = NULL)

for (i in 1:length(x)) {
  for (j in 1:length(y)) {
    z[i,j] = compute_logistics(w0_f, w1_f, w2_f, x[i], y[j])
  }
}

plot3 <- plot_ly(x = x, y = y, z = z, type = 'mesh3d') %>% add_surface() %>% layout(scene = list(xaxis = list(title = "x1"), yaxis = list(title = "x2")), title = "Neural Network Output")

mean_square_error <- function(data, class, w0, w1, w2) {
  result <- rep(0, nrow(data))
  for (i in 1:nrow(data)) {
    result[i] = compute_logistics(w0, w1, w2, data$petal_length[i], data$petal_width[i])
    result[i] = (result[i] - class[i])^2
  }
  mse = mean(result)
  return (mse)
}

d <- plotdata %>% select(petal_length, petal_width)
c <- rep(0, nrow(d))

for (i in 1:nrow(d)) {
  if (plotdata$species[i] == 'virginica')
    c[i] = 1
}
mean_square_error(d, c, w0_f, w1_f, w2_f)
mean_square_error(d, c, 1, 2, 3)

plot4 <- plot1 + stat_function(fun = boundary, args = list(w0 = w0_f, w1 = w1_f, w2 = w2_f))
plot5 <- plot1 + stat_function(fun = boundary, args = list(w0 = 1, w1 = 2, w2 = 3))

gradient_mse <- function(data, class, w0, w1, w2) {
  dMSE_dw <- c(0, 0, 0)
  for (i in 1:nrow(data)) {
    x_i1 = data$petal_length[i]
    x_i2 = data$petal_width[i]
    c_i = class[i]
    
    y_i = compute_logistics(w0, w1, w2, x_i1, x_i2)
    
    dMSE_dw[1] = dMSE_dw[1] + (2/nrow(data)) * (y_i-c_i) * y_i * (1-y_i)
    dMSE_dw[2] = dMSE_dw[2] + (2/nrow(data)) * (y_i-c_i) * y_i * (1-y_i) * x_i1
    dMSE_dw[3] = dMSE_dw[3] + (2/nrow(data)) * (y_i-c_i) * y_i * (1-y_i) * x_i2
  }
  return (dMSE_dw)
}

w <- c(1, 2, 3)
print(paste("MSE: ", mean_square_error(d, c, w[1],w[2],w[3])))
plot6 <- plot1 + stat_function(fun = boundary, args = list(w0 = w[1], w1 = w[2], w2 = w[3]))

for (i in 1:35) {
  w <- w - 500 * gradient_mse(d, c, w[1], w[2], w[3])
}
print(paste("MSE: ", mean_square_error(d, c, w[1],w[2],w[3])))
plot6 <- plot1 + stat_function(fun = boundary, args = list(w0 = w[1], w1 = w[2], w2 = w[3]))

for (i in 1:5000) {
  w <- w - 1 * gradient_mse(d, c, w[1], w[2], w[3])
}
print(paste("MSE: ", mean_square_error(d, c, w[1],w[2],w[3])))
plot6 <- plot1 + stat_function(fun = boundary, args = list(w0 = w[1], w1 = w[2], w2 = w[3]))

for (i in 1:20000) {
  w <- w - 0.002 * gradient_mse(d, c, w[1], w[2], w[3])
}
print(paste("MSE: ", mean_square_error(d, c, w[1],w[2],w[3])))
plot6 <- plot1 + stat_function(fun = boundary, args = list(w0 = w[1], w1 = w[2], w2 = w[3]))

optimizer <- function(w0, w1, w2, step, numIteration) {
  w <- c(w0, w1, w2)
  mse <- rep(0, numIteration)
  ite <- seq(1, numIteration, by = 1)
  
  for (i in 1:numIteration) {
    if (i == 1)
      prev <- c(0, 0, 0)
    
    if (abs(sum(w)- sum(prev)) < 0.0001)
      break;
    
    mse[i] = mean_square_error(d, c, w[1], w[2], w[3])
    prev <- w
    w <- w - step * gradient_mse(d, c, w[1], w[2], w[3])
  }
  
  print(w)
  print(paste("MSE: ", mean_square_error(d, c, w[1],w[2],w[3])))
  
  plot_a <- plot1 + stat_function(fun = boundary, args = list(w0 = w[1], w1 = w[2], w2 = w[3]))
  plot_b <- ggplot(data.frame(ite, mse), aes(x = ite, y = mse)) + geom_point()
  ggarrange(plot_a, plot_b, ncol = 1, nrow = 2, heights = c(3,2), align = "hv")
}

randW0 <- runif(1, min=-5, max=0)
randW1 <- runif(1, min=0, max=1)
randW2 <- runif(1, min=0, max=5)
optimizer(randW0, randW1, randW2, 0.005, 10)
optimizer(randW0, randW1, randW2, 0.005, 100)
optimizer(randW0, randW1, randW2, 0.005, 500)

