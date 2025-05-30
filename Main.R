
# setwd("C:/Users/jafio/OneDrive - unb.br/Pesquisas/Energia - Marco Machado/Code")


source("Libraries.R")
source("RollingWindowFunctions.R")
# source("DescritiveAnalisys.R")


# ------------ Dataset ---------------------------------------------------------
# load the dataset
df <- read.csv("PreProcessedData.csv")
df$datetime <- as.POSIXct(df$datetime , format = "%Y-%m-%d %H:%M:%S") 
x <- xts(df[,-1], order.by = df$datetime)
rm(df)


# to fit the models
n1 = 78888
treino_W <- msts( x[1:n1], seasonal.periods=c(24*7) )
treino_DW <- msts( x[1:n1], seasonal.periods=c(24, 24*7) )
treino_DWY <- msts( x[1:n1], seasonal.periods=c(24, 24*7, 365.25*24) )

# to evaluations
x_W <- msts( x, seasonal.periods=c(24*7) )
x_DW <- msts( x, seasonal.periods=c(24, 24*7) )
x_DWY <- msts( x, seasonal.periods=c(24, 24*7, 365.25*24) )

rm(x)
#' -----------------------------------------------------------------------------


# --------- Selection and fitting all models -----------------------------------


## ---- holt winters ----
hw_W = HoltWinters(treino_W, optim.start = c(alpha = 0.01, beta = 0.01, gamma = 0.01))

hw_DW = dshw(treino_DW)#$model
##


## ---- tbats ----
tempo1 = Sys.time()
tbats_W = tbats(treino_W, num.cores = NULL, use.parallel = T)
print( Sys.time() - tempo1) # Time difference of 8.062259 mins

tempo1 = Sys.time()
tbats_DW = tbats(treino_DW, num.cores = NULL, use.parallel = T)
print( Sys.time() - tempo1) # Time difference of 20.96187 mins

tempo1 = Sys.time()
tbats_DWY = tbats(treino_DWY, num.cores = NULL, use.parallel = T)
print( Sys.time() - tempo1) # Time difference of 29.9484 mins
##


## ---- SARIMA-W  ---- 

# treino_W %>% ndiffs()
# treino_W %>% diff() %>% nsdiffs()
tempo = Sys.time()
sarima_W = auto.arima(treino_W, d=1, D=1,
                      max.P = 1, max.Q = 1,
                      allowdrift = F,
                      trace = T
                      )
# # Best model: ARIMA(4,1,3)(0,1,0)[168]
tempo2 = Sys.time()
#  
# tempo2-tempo
# Time difference of 1.52617 hours

sarima_W = Arima(treino_W, order=c(4,1,3), seasonal=c(0,1,0))
sarima_W2 = msarima(treino_W, orders=list(ar=c(4,0), i=c(1,1),ma=c(3,0)),lags=c(1,168)) # fast for rolling window
## --- 


## ---- Selection of SARIMA-DW  ---- 
e = sarima_W$residuals %>% as.numeric() %>% ts(frequency = 24)
e %>% nsdiffs() ## nao precisa de diferencas sazonais de ciclo 24
acf(e, lag.max=5*24)
pacf(e, lag.max=5*24)
## ARIMA(4,1,3)[1](1,0,1)[24](0,1,0)[168]
# sarima_DW = msarima(treino_DW, orders=list(ar=c(4,1,0), i=c(1,0,1),ma=c(3,1,0)),lags=c(1,24,168))
# e = sarima_DW$residuals  %>% ts(frequency = 24)
# acf(e, lag.max=5*24)
# pacf(e, lag.max=5*24)

## ARIMA(4,1,3)[1](1,0,0)[24](0,1,0)[168]
#sarima_DW1 = msarima(treino_DW, orders=list(ar=c(4,1,0), i=c(1,0,1),ma=c(3,0,0)),lags=c(1,24,168))
# Information criteria:
#   AIC    AICc     BIC    BICc
# 1180176 1180176 1180259 1180259


## ARIMA(4,1,3)[1](2,0,0)[24](0,1,0)[168]
#sarima_DW2 = msarima(treino_DW, orders=list(ar=c(4,2,0), i=c(1,0,1),ma=c(3,0,0)),lags=c(1,24,168))
# Information criteria:
#   AIC    AICc     BIC    BICc 
# 1179538 1179538 1179631 1179631 

## ARIMA(4,1,3)[1](1,0,1)[24](0,1,0)[168]
sarima_DW = msarima(treino_DW, orders=list(ar=c(4,1,0), i=c(1,0,1),ma=c(3,1,0)),lags=c(1,24,168))
# Information criteria:
#   AIC    AICc     BIC    BICc 
# 1179436 1179436 1179529 1179529

e = sarima_DW$residuals  %>% ts(frequency = 24)
acf(e, lag.max=5*24)
pacf(e, lag.max=5*24)

sarima_DW$timeElapsed
## Time difference of 15.99891 mins

## ---


## ---- Harmonic Dinamic Regression ---- 

tempo1 = Sys.time()
s_harm_W = fourier(treino_W, K=c(7))
hr_W = auto.arima(treino_W, lambda="auto", seasonal=F, xreg=s_harm_W)
print( Sys.time() - tempo1) # Time difference of 4.645008 mins

tempo1 = Sys.time()
s_harm_DW = fourier(treino_DW, K=c(7,7))
hr_DW = auto.arima(treino_DW, lambda="auto", seasonal=F, xreg=s_harm_DW)
print( Sys.time() - tempo1) # Time difference of 8.68496 mins

tempo1 = Sys.time()
s_harm_DWY = fourier(treino_DWY, K=c(7,7,7))
hr_DWY = auto.arima(treino_DWY, lambda="auto", seasonal=F, xreg=s_harm_DWY)
print( Sys.time() - tempo1) # Time difference of 36.10139 mins
##  --- 


## ---- LSTM ---- 
tempo = Sys.time()

lstm_W = f_lstm( y=treino_W, h=NULL, seas="W" )$model
save_model_hdf5(object=lstm_W, filepath='modelos_treinados/lstm_W.h5')

lstm_DW = f_lstm( y=treino_DW, h=NULL, seas="DW" )$model
save_model_hdf5(object=lstm_DW, filepath='modelos_treinados/lstm_DW.h5')

lstm_DWY =f_lstm( y=treino_DWY, h=NULL, seas="DWY" )$model
save_model_hdf5(object=lstm_DWY, filepath='modelos_treinados/lstm_DWY.h5')

tempo_fit_lstm = Sys.time() - tempo
tempo_fit_lstm
## ---

#'-----------------------------------------------------------------------------



# --------- Run Rolling Window Evaluation --------------------------------------

initial = 7888 




##  ----- Holt Winters  ----- 
t_hw_W = Sys.time()
CV_hw_W = fast_tsCV(x_W, forecastfunction=f_hw, h=168, 
                    initial=initial, model=hw_W, 
                    arq_log="log_hw_W.txt")
t_hw_W =  Sys.time() - t_hw_W ## Time difference of 15.08899 mins
save.image("CV1.RData")


t_hw_DW = Sys.time()
CV_hw_DW = fast_tsCV(x_DW, forecastfunction=f_dshw, h=168, 
                     initial=initial, model=hw_DW, 
                     arq_log="log_hw_DW.txt")
t_hw_DW =  Sys.time() - t_hw_DW ## Time difference of 1.877418 hours
save.image("CV2.RData")
##'

## ----- TBATS  ----- 
t_tbats_W = Sys.time()
CV_tbats_W = fast_tsCV(x_W, forecastfunction=f_tbats, h=168, 
                       initial=initial, model=tbats_W, 
                       arq_log="log_tbats_W.txt")
t_tbats_W =  Sys.time() - t_tbats_W ## Time difference of 12.39654 mins
save.image("CV3.RData")


t_tbats_DW = Sys.time()
CV_tbats_DW = fast_tsCV(x_DW, forecastfunction=f_tbats, h=168, 
                        initial=initial, model=tbats_DW, 
                        arq_log="log_tbats_DW.txt")
t_tbats_DW =  Sys.time() - t_tbats_DW ## Time difference of 13.25661 mins
save.image("CV4.RData")


t_tbats_DWY = Sys.time()
CV_tbats_DWY = fast_tsCV(x_DWY, forecastfunction=f_tbats, h=168, 
                         initial=initial, model=tbats_DWY, 
                         arq_log="log_tbats_DWY.txt")
t_tbats_DWY =  Sys.time() - t_tbats_DWY ## Time difference of 18.43076 mins
save.image("CV5.RData")
##'

##  ----- ARIMA  ----- 

t_snaive = Sys.time()
CV_snaive_W = fast_tsCV(x_W, forecastfunction=snaive, h=168, 
                        initial=initial, 
                        arq_log="log_snaive_W.txt")
t_snaive =  Sys.time() - t_snaive   ## Time difference of 8.340908 mins
t_snaive
save.image("CV6.RData")

t_sarima_W = Sys.time()
CV_sarima_W = fast_tsCV(x_W, forecastfunction=f_msarima, h=168, 
                        initial=initial, 
                        model=sarima_W2, 
                        arq_log="log_sarima_W.txt")
t_sarima_W =  Sys.time() - t_sarima_W ## Time difference of 13.90473 hours
save.image("CV7.RData")

t_sarima_DW = Sys.time()
CV_sarima_DW = fast_tsCV(x_DW, forecastfunction=f_msarima, h=168, 
                         initial=initial, 
                         model=sarima_DW,
                         window_fit = 40000,
                         arq_log="log_sarima_DW.txt")

t_sarima_DW =  Sys.time() - t_sarima_DW ## Time difference of 1.027161 days
save.image("CV8.RData")
##'


## ----- Harmonic Regression  ----- 
s_harm_W = fourier(x_W, K=c(7))
s_harm_DW = fourier(x_DW, K=c(7,7))
s_harm_DWY = fourier(x_DWY, K=c(7,7,7))


t_hr_W = Sys.time()
CV_hr_W = fast_tsCV(x_W, forecastfunction=f_arima, h=168, 
                    initial=initial, model=hr_W, xreg=s_harm_W )
t_hr_W =  Sys.time() - t_hr_W ## Time difference of 31.91947 mins
save.image("CV9.RData")


t_hr_DW = Sys.time()
CV_hr_DW = fast_tsCV(x_DW, forecastfunction=f_arima, h=168, 
                     initial=initial, model=hr_DW, xreg=s_harm_DW )
t_hr_DW =  Sys.time() - t_hr_DW ## Time difference of 42.38414 mins
save.image("CV10.RData")


t_hr_DWY = Sys.time()
CV_hr_DWY = fast_tsCV(x_DWY, forecastfunction=f_arima, h=168, 
                      initial=initial, model=hr_DWY, xreg=s_harm_DWY )
t_hr_DWY =  Sys.time() - t_hr_DWY ## Time difference of 59.42863 mins
save.image("CV11.RData")
##'





##  ----- NNETAR  ----- 
## omitted from the article because it does not work with multiple seasonality and has low performance.
t_nnetar_W = Sys.time()
CV_nnetar_W = fast_tsCV(x_W, forecastfunction=f_nnetar, h=168, 
                        initial=initial, model=nnetar_W, 
                        arq_log="log_nnetar_W.txt")
t_nnetar_W =  Sys.time() - t_nnetar_W ## Time difference of 4.687043 hours
save.image("CV12.RData")


t_nnetar_DW = Sys.time()
CV_nnetar = fast_tsCV(x_DW, forecastfunction=f_nnetar, h=168, 
                      initial=initial, model=nnetar_DW, 
                      arq_log="log_nnetar_DW.txt")
t_nnetar_DWY =  Sys.time() - t_nnetar_DW ## Time difference of 4.283874 hours
save.image("CV13.RData")


t_nnetar_DWY = Sys.time()
CV_nnetar = fast_tsCV(x_DWY, forecastfunction=f_nnetar, h=168, 
                      initial=initial, model=nnetar_DWY, 
                      arq_log="log_nnetar_DWY.txt")
t_nnetar_DWY =  Sys.time() - t_nnetar_DWY ## Time difference of 4.283874 hours
save.image("CV14.RData")
##



##  ----- LSTM  ----- 

## Approximately 50 hours - Windows 11, Laptop I7 9750h
cat("Total de avaliações: ", length(x_W) - initial)
lstm_W = load_model_hdf5(filepath='modelos_treinados/lstm_W.h5')
t_lstm_W = Sys.time()
CV_lstm_W = fast_tsCV(x_W, forecastfunction=f_lstm, h=168, 
                      initial=initial, model=lstm_W,
                      seas = "W", arq_log="log_lstm_W_.txt")
t_lstm_W =  Sys.time() - t_lstm_W  
t_lstm_W              ## Time difference of 1.264964 days
save.image("CV15.RData")
##'

## Approximately 50 hours - Windows 11, Laptop I7 9750h
cat("Total de avaliações: ", length(x_DW) - initial)
lstm_DW = load_model_hdf5(filepath='modelos_treinados/lstm_DW.h5')
t_lstm_DW = Sys.time()
CV_lstm_DW = fast_tsCV(x_DW, forecastfunction=f_lstm, h=168, 
                       initial=initial, model=lstm_DW,
                       seas = "DW", arq_log="log_lstm_DW.txt")
t_lstm_DW =  Sys.time() - t_lstm_DW
t_lstm_DW            ## Time difference of 1.25187 days
save.image("CV16.RData")
##'

## Approximately 50 hours - Windows 11, Laptop I7 9750h
cat("Total de avaliações: ", length(x_DWY) - initial)
lstm_DWY = load_model_hdf5(filepath='modelos_treinados/lstm_DWY.h5')
t_lstm_DWY = Sys.time()
CV_lstm_DWY = fast_tsCV(x_DWY, forecastfunction=f_lstm, h=168, 
                        initial=initial, model=lstm_DWY,
                        seas = "DWY", arq_log="log_lstm_DWY.txt")
t_lstm_DWY =  Sys.time() - t_lstm_DWY  
t_lstm_DWY          ## Time difference of 1.579664 days
save.image("CV17.RData")
##'


#' -----------------------------------------------------------------------------









# -------- Results -------------------------------------------------------------
load("CV_complete.RData")

models_names = c(
  "SNAIVE-W"
  ,"SARIMA-W"
  ,"SARIMA-DW"
  ,"HR-W"
  ,"HR-DW"
  ,"HR-DWY"
  ,"HW-W"
  ,"HW-DW"
  ,"TBATS-W"
  ,"TBATS-DW"
  ,"TBATS-DWY"
  #,"NNETAR"
  ,"LSTM-W"
  ,"LSTM-DW"
  ,"LSTM-DWY")

MAPE_snaive_W = CV_snaive_W %>% abs() %>% colMeans(na.rm=T)
MAPE_sarima_W = CV_sarima_W %>% abs() %>% colMeans(na.rm=T)
MAPE_sarima_DW = CV_sarima_DW %>% abs() %>% colMeans(na.rm=T)

MAPE_hr_W = CV_hr_W %>% abs() %>% colMeans(na.rm=T)
MAPE_hr_DW = CV_hr_DW %>% abs() %>% colMeans(na.rm=T)
MAPE_hr_DWY = CV_hr_DWY %>% abs() %>% colMeans(na.rm=T)

MAPE_hw_W = CV_hw_W %>% abs() %>% colMeans(na.rm=T)
MAPE_hw_DW = CV_hw_DW %>% abs() %>% colMeans(na.rm=T)

MAPE_tbats_W = CV_tbats_W %>% abs() %>% colMeans(na.rm=T)
MAPE_tbats_DW = CV_tbats_DW %>% abs() %>% colMeans(na.rm=T)
MAPE_tbats_DWY = CV_tbats_DWY %>% abs() %>% colMeans(na.rm=T)

MAPE_nnetar = CV_nnetar %>% abs() %>% colMeans(na.rm=T)

MAPE_lstm_W = CV_lstm_W %>% abs() %>% colMeans(na.rm=T)
MAPE_lstm_DW = CV_lstm_DW %>% abs() %>% colMeans(na.rm=T)
MAPE_lstm_DWY = CV_lstm_DWY %>% abs() %>% colMeans(na.rm=T)





## ---- MAPE by hours   ----- 
tab = 100*cbind( 
  MAPE_snaive_W,           
  MAPE_sarima_W, 
  MAPE_sarima_DW,
  MAPE_hr_W,
  MAPE_hr_DW,
  MAPE_hr_DWY,
  MAPE_hw_W,
  MAPE_hw_DW,
  MAPE_tbats_W,
  MAPE_tbats_DW,
  MAPE_tbats_DWY,
  #MAPE_nnetar,
  MAPE_lstm_W,
  MAPE_lstm_DW,
  MAPE_lstm_DWY
)

colnames(tab) = models_names
##




## --- MAPE by days   ----- 
tab2 = NULL
nomes = NULL
for(dia in 1:7){
  horas = (24*(dia-1)+1):(24*(dia))
  tab2 = rbind( tab2, colMeans(tab[horas, ]) )
  nomes = c(nomes, paste0(horas[1],"-",horas[24]) )
  print(horas)
}
row.names(tab2) = nomes
##


print(tab)

max(tab, na.rm=T)




## ----- Graphics ----- 

subgrupos = list()
subgrupos[["SARIMA"]] = c("SNAIVE-W","SARIMA-W", "SARIMA-DW")
subgrupos[["HR"]] = c("HR-W", "HR-DW", "HR-DWY")
subgrupos[["HW"]] = c("HW-W", "HW-DW")
subgrupos[["TBATS"]] = c("TBATS-W", "TBATS-DW", "TBATS-DWY")
subgrupos[["LSTM"]] = c("LSTM-W", "LSTM-DW", "LSTM-DWY")
subgrupos[["Best24"]] = c("SARIMA-DW", "HR-DWY", "HW-DW", "TBATS-W", "LSTM-DWY")
subgrupos[["Best168"]] = c("SNAIVE-W", "HR-DW", "HW-DW", "TBATS-W", "LSTM-W")
subgrupos[["Best"]] = c("SARIMA-DW", "TBATS-W", "LSTM-W")

i=1
for( family in names(subgrupos)){ 
  
  # family = "Best"
  models = subgrupos[[family]]
  m = length(models)
  
  
  #
  pdf( file = paste0("imagens//24hours_", family,".pdf"), width =9, height = 7 )
  
  plot.ts(tab[1:24,models], 
          ylim=c(0, 10),
          xlim=c(0, 24),
          plot.type='s',
          col=1:m, 
          lwd=rep(3,m), 
          xlab="hour", 
          ylab="MAPE(%)",
          main=paste0("(",letters[i],")")
  )
  legend(x=0, y=10, 
         legend=models, 
         col=1:m, lwd=rep(3,m) )
  
  dev.off()
  #
  
  
  
  
  #
  pdf( file = paste0("imagens//hour_", family,".pdf"), width =9, height = 7 )
  
  plot.ts(tab[,models], 
          ylim=c(0, 15), 
          plot.type='s',
          col=1:m, 
          lwd=rep(3,m), 
          xlab="hour", 
          ylab="MAPE(%)",
          main=paste0("(",letters[i],")")
  )
  legend(x=0, y=15, 
         legend=models, 
         col=1:m, lwd=rep(3,m) )
  
  dev.off()
  #
  
  
  #
  pdf( file = paste0("imagens//day_", family,".pdf"), width =9, height = 7)
  
  plot.ts(tab2[,models], 
          ylim=c(0, 15), 
          plot.type='s',
          col=1:m, 
          lwd=rep(3,m), 
          xlab="day", 
          ylab="MAPE(%)",
          main=paste0("(",letters[i],")")
  )
  legend(x=1, y=15, 
         legend=models, 
         col=1:m, lwd=rep(3,m) )
  
  dev.off()
  #
  
  i = i+1
}

## 


write.csv(round(rbind(tab,tab2),2), file = "resultados.csv")



## -----  Table first hours   ----- 
tab_ = tab[1:5, ]
names_ = c( row.names(tab_), "1-24" )
tab_ = rbind(tab_, colMeans(tab[1:24,]))
row.names(tab_) = names_  

tab_ = tab_ %>% round(3) %>% t()

write.csv(round(tab_,3), file = "resultados_tab_primeiras.csv")



##   ----- Table of days means   ----- 
tab_mean = matrix(NA, nrow=8, ncol=ncol(tab))
row.names(tab_mean) = c("1-24","25-48","49-72","73-96","97-120","121-144","145-168","1-168")
colnames(tab_mean) = colnames(tab)

tab_mean[1,] = tab[1:24,] %>% colMeans()
tab_mean[2,] = tab[25:48,] %>% colMeans()
tab_mean[3,] = tab[49:72,] %>% colMeans()
tab_mean[4,] = tab[73:96,] %>% colMeans()
tab_mean[5,] = tab[97:120 ,] %>% colMeans()
tab_mean[6,] = tab[121:144 ,] %>% colMeans()
tab_mean[7,] = tab[145:168 ,] %>% colMeans()
tab_mean[8,] = tab[1:168,] %>% colMeans()

tab_mean = tab_mean %>% round(3) %>% t()

write.csv(round(tab_mean,3), file = "resultados_tab_mean.csv")


#### classificacao 1:24

## por familia 
aux = tab[1:24,subgrupos[["Best24"]]] %>% colMeans()
rank(aux)

##
tab[1:24,] %>% colMeans() %>% rank() %>% sort()



## ---- Diebold-Mariano test for predictive accuracy ----- 

### H0: Acuracy M1 = Acuracy M2
### H1: Acuracy M1 > Acuracy M2

p_values = matrix(NA, nrow=length(models_names), ncol=length(models_names))
row.names(p_values) = colnames(p_values) = models_names

h=1
hh = paste0("h=", h)
results_hh = cbind(
                  CV_snaive_W[,hh],           
                  CV_sarima_W[,hh], 
                  CV_sarima_DW[,hh],
                  CV_hr_W[,hh],
                  CV_hr_DW[,hh],
                  CV_hr_DWY[,hh],
                  CV_hw_W[,hh],
                  CV_hw_DW[,hh],
                  CV_tbats_W[,hh],
                  CV_tbats_DW[,hh],
                  CV_tbats_DWY[,hh],
                  CV_lstm_W[,hh],
                  CV_lstm_DW[,hh],
                  CV_lstm_DWY[,hh]
                  )

results_hh = tail(results_hh, 1000)

colnames(results_hh) = models_names

for(i in models_names){
  for(j in models_names){
    
    if(i == j) next;
    
    out = dm.test(
      e1=results_hh[,i],
      e2=results_hh[,j],
      alternative = c("less"),
      h = h
    )
    
    p_values[i,j] = round(out$p.value,2)
  }
}


print( round(p_values,4) )


write.csv( data.frame( format(p_values, nsmall=2) ) , file = "DM_pvalues.csv")

## ---


## ---- Computational time ----- 
tempos = c( 
  t_snaive,           
  t_sarima_W, 
  t_sarima_DW,
  t_hr_W,
  t_hr_DW,
  t_hr_DWY,
  t_hw_W,
  t_hw_DW,
  t_tbats_W,
  t_tbats_DW,
  t_tbats_DWY,
  #t_nnetar,
  t_lstm_W,
  t_lstm_DW,
  t_lstm_DWY
)

names(tempos) = models_names

tempos_df = data.frame( round(tempos,0))

colnames(tempos_df) = c("Computational Time")

write.csv( data.frame( format(tempos_df , nsmall=2) ) , file = "computational_time.csv")

## ---

#' -----------------------------------------------------------------------------
