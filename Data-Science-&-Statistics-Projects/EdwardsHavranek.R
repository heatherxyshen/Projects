## Here we replicate the model selection process of Edwards and Havranek (1985) used in the Reinis dataset contingency table

library(gRbase)
library(gRim)
data(reinis)
str(reinis)

## Start with the saturated model
m.init <- dmod(~.^., data=reinis)

## Step 1
testdelete(m.init, ~smoke:mental)
testdelete(m.init, ~smoke:phys)
testdelete(m.init, ~smoke:systol)
testdelete(m.init, ~smoke:protein)
testdelete(m.init, ~smoke:family)

testdelete(m.init, ~mental:phys)
testdelete(m.init, ~mental:systol)
testdelete(m.init, ~mental:protein)
testdelete(m.init, ~mental:family)

testdelete(m.init, ~phys:systol)
testdelete(m.init, ~phys:protein)
testdelete(m.init, ~phys:family)

testdelete(m.init, ~systol:protein)
testdelete(m.init, ~systol:family)

testdelete(m.init, ~protein:family)

## Step 2
comparemodels <- function(m1,m2) {
  lrt <- m2$fitinfo$dev - m1$fitinfo$dev
  dfdiff <- m1$fitinfo$dimension[1] - m2$fitinfo$dimension[1]
  c('lrt'=lrt, 'df'=dfdiff)
}

m.2 <- dmod(~smoke:phys + smoke:systol + smoke:protein + mental:phys 
            + systol:protein, data=reinis, details = 1)
1 - pchisq(m.2$fitinfo$dev, m.2$fitinfo$dimension[1])

## Step 3
testadd(m.2, ~smoke:mental, details = 1)
testadd(m.2, ~smoke:family, details = 1)
testadd(m.2, ~mental:systol, details = 1)
testadd(m.2, ~mental:protein, details = 1) #*
testadd(m.2, ~mental:family, details = 1)
testadd(m.2, ~phys:systol, details = 1)
testadd(m.2, ~phys:protein, details = 1) #*
testadd(m.2, ~phys:family, details = 1)
testadd(m.2, ~systol:protein, details = 1)
testadd(m.2, ~systol:family, details = 1)

m.3.1 = dmod(~smoke:phys + smoke:systol + smoke:protein + mental:phys 
             + systol:protein + mental:protein + family, data=reinis, details = 1)
m.3.2 = dmod(~smoke:phys + smoke:systol + smoke:protein + mental:phys 
             + systol:protein + phys:protein + family, data=reinis, details = 1)
## Step 4
m.4.1 <- update(m.3.1, list(dedge= ~ mental:protein + phys:protein))
m.4.2 <- update(m.3.2, list(dedge= ~ mental:protein + phys:protein))
1 - pchisq(m.4.1$fitinfo$dev, m.4.1$fitinfo$dimension[1])
1 - pchisq(m.4.2$fitinfo$dev, m.4.2$fitinfo$dimension[1])

## Final Model
plot(m.3.1)
dev.off()
plot(m.3.2)
