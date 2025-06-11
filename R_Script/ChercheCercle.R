# Réalisé par Gilles le Moguédec

coords<-read.csv("coords.csv", header=T)
# Soit un ensemble de points (x,y)
# On cherche le cercle qui passe "au mieux" parmi ces points

# Distance au carré entre un point (x,y) et un cercle de centre (x0,y0) et de rayon R
fD2<-function(x,y,x0,y0,R){
  D2<-(sqrt( (x-x0)^2 + (y-y0)^2) - R)^2
  return(D2)
}

# Fonction à optimiser : somme des distances au carré
lsq<-function(Par,x,y){
  x0<-Par[1]
  y0<-Par[2]
  R<-Par[3]
  D2<-fD2(x=x,y=y,x0=x0,y0=y0,R=R)
  return(sum(D2,na.rm = TRUE))
}

# generation de points avec bruitage
generPts<-function(n=1,x0=0,y0=0,R=1,Sigma=1){
  Theta<-2*pi*runif(n=n)
  x<-R*cos(Theta) + Sigma*rnorm(n=n)
  y<-R*sin(Theta) + Sigma*rnorm(n=n)
  Sortie<-cbind(x,y)
  return(Sortie)
}

# Pour valeur initiale des paramètres
# Si x0 ou y0 n'est pas précisé, on prend la moyenne des x ou des y
fInit<-function(x,y,x0=NA,y0=NA){
  if (is.na(x0)){ x0<-mean(x,na.rm=TRUE)}
  if (is.na(y0)){ y0<-mean(y,na.rm=TRUE)}
  R2<- (x-x0)^2 + (y-y0)^2
  R <- mean( sqrt(R2))
  Sortie<-c(x0=x0,y0=y0,R=R)
  return(Sortie)
}


# Graphique des résultats
graphResult<-function(Tableau,Param=NULL,Start=NULL){
  xlim<-range(Tableau[,"x"],na.rm=TRUE)
  if (!is.null(Param)){ xlim<-range(xlim,Param["x0"])}
  ylim<-range(Tableau[,"y"],na.rm=TRUE)
  if (!is.null(Param)){ ylim<-range(ylim,Param["y0"])}
  
  plot(x=Tableau,xlim=xlim,ylim=ylim,xlab="x",ylab="y",pch=19,col="blue",asp=1)
  Theta<-seq(0,2*pi,length=1001)
  if (!is.null(Start)){
    lines(x=Start["R"]*cos(Theta)+Start["x0"], y=Start["R"]*sin(Theta)+Start["y0"], col="grey",lty=2)
    points(x=Start["x0"], y=Start["y0"], col="grey")
  }
  if (!is.null(Param)){
    lines(x=Param["R"]*cos(Theta)+Param["x0"], y=Param["R"]*sin(Theta)+Param["y0"], col="red")
    points(x=Param["x0"], y=Param["y0"], col="red")
  }
}
########### Application

#Tableau<-generPts(n=5,x0=1,y0=2,R=1,Sigma=.1)
Tableau<-list(x=coords$X,y=coords$Y)
Tableau<-cbind(x=coords$X,y=coords$Y)
Start<-fInit(x=Tableau[,"x"],y=Tableau[,"y"])
Start

# Recherche du meilleur cercle
Optim<-optim(fn=lsq,par = Start, x=Tableau[,"x"],y=Tableau[,"y"] )
print(Optim)

graphResult(Tableau = Tableau, Param=Optim$par, Start=Start)


