sample(1:6, 6, replace = F)
#21 feet away
#moving board closer, counting hitting the board as on the board, cutting out projectile object

#Katina 6, 1, 3, 5, 2, 4
  #non-dom, lacrosse
    #3 went in, 0, 1 hit board, 0
  #dom, beanbag
    #1 hit board, 0, 1 hit board, 1 hit board
  #dom, tennis ball
    #0, 0, 0, 0
  #dom, lacrosse
    #0, 0, 3, 1
  #non-dom, beanbag
    #0, 0, 0, 0
  #non-dom, tennis
    #0, 1, 1, 1
#Sara 2, 4, 1, 6, 5, 3
  #non-dom, beanbag
    #0, 0, 1, 1
  #non-dom, tennis ball
    #1, 1, 1, 0
  #dom, beanbag
    #1, 1, 3, 1
  #non-dom, lacrosse
    #0, 1, 0, 1
  #dom, lacrosse
    #1, 1, 1, 1
  #dom, tennis
    #0, 1, 1, 0
#Heather 6, 2, 1, 5, 3, 4
  #non-dom, lacrosse
    #0, 0, 1, 0
  #non-dom, beanbag
    #1, 0, 0, 1
  #dom, beanbag
    #0, 1, 0, 1
  #dom, lacrosse
    #0, 0, 1, 1

data_ch = matrix(c(2, 3, 6, 2, 0, 2, 1, 0, 2, 2, 3, 3, 2, 4, 4, 1, 4, 2), nrow = 6, byrow = T)
row.names(data_ch) = c("bb, D", "bb, nD", "t, D", "t, nD", "l, D", "l, nD")
colnames(data_ch) = c("H", "K", "S")
var_ch = apply(data_ch, 1, var)
mean_ch = apply(data_ch, 1, mean)
sigma_sqrd = var(c(2, 3, 6, 2, 0, 2, 1, 0, 2, 2, 3, 3, 2, 4, 4, 1, 4, 2))/(18-1)
scores = c(2, 3, 6, 2, 0, 2, 1, 0, 2, 2, 3, 3, 2, 4, 4, 1, 4, 2)
#Power
mean_hatch = mean(mean_ch)
alpha = 0.05
m = 6
n = 3
N = n*m
mse_ch = sum((scores - mean_hatch)^2)/12
f.crit = qf(1-alpha, m - 1, N - m)
f.ncp = n*sum((mean_ch - mean_hatch)^2)/mse_ch
f.power = 1 - pf(f.crit, m-1, N-m, ncp=f.ncp)
#sample size
n.2 = seq(2, 250)
m.2 = 6
N.2 = n.2*m.2
f.crit.2 = qf(1-alpha, m.2 - 1, N.2 - m.2)
f.ncp.2 = numeric(249)
for(i in 1:249) {
  f.ncp.2[i] = n.2[i]*sum((mean_ch - mean_hatch)^2)/mse_ch
}
f.power.2 = 1 - pf(f.crit.2, m.2-1, N.2-m.2, ncp=f.ncp.2)
n.2[which(f.power.2 > 0.9)]
