
# forecast functions


## Rolling Window function 
fast_tsCV = function( x, xreg=NULL, forecastfunction=f_arima, h=168, 
                      initial=96400, window_fit=NULL, model=sarima_W, arq_log=NULL, 
                      ... ){
  
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  clusterExport(cl, c("forecast",
                      "Arima","ets","msarima","tbats","hw","dshw",
                      "nnetar", "predict"))
  
  escreve_log = function(texto, arq_name){
    Sys.setenv(TZ="America/Sao_Paulo")
    sink(arq_name, append=TRUE)
    a=try( cat( paste(Sys.time(),":", texto , "\n" ) ) )
    sink()
  }

  
  n = length(x)
  
  m = foreach(ni = initial:(n-1), .combine = rbind, .packages=c("keras") )  %do% {
    
    if(!is.null(arq_log) && ((ni %% 10)==0)){
      aux = paste("ni = ", toString(ni) )
      escreve_log( aux , arq_name = arq_log )
    }
    
    if(is.null(window_fit)){
      treino <- window( x, end=time(x)[ni] )
    }else{
      treino <- window( x, start=time(x)[ni-window_fit], end=time(x)[ni] )
    }
    
    if(ni+h > n){
      h_ = n - ni
    }else{
      h_ = h
    }
    teste <- as.numeric( x[(ni+1):(ni+h_)] )
    
    if(is.null(xreg)){
      f = try( forecastfunction( treino, h=h_, model=model, ... )$mean )
    }else{
      treino_reg <- xreg[1:ni,,drop=F]
      test_reg <- xreg[(ni+1):(ni+h_),,drop=F]
      f = try( forecastfunction( treino, h=h_, xreg=treino_reg, newxreg=test_reg, 
                                 model=model, ... )$mean )
    }
    
    if("try-error" %in% class(f)){
      e = rep(NA, h)  
    }else{
      e = c( (teste-as.numeric(f))/teste, rep(NA,h-h_) )
    }
    
    if(!is.null(arq_log) && ((ni %% 10)==0) ){
      aux = paste("ni = ", toString(ni), "Terminou" )
      escreve_log( aux , arq_name = arq_log )
    }
    
    return(e)
  }
  
  stopCluster(cl)
  
  colnames(m) = paste0("h=",1:h)
  
  return(m)
}




## SARIMA and Dinamic Harmonic Regression
f_arima <- function( y, h, xreg=NULL, newxreg=NULL, model=NULL ){
  fit = Arima( y, xreg=xreg, model=model )
  forecast( fit, h=h, xreg=newxreg )
}

## Multi Seasonal SARIMA
f_msarima <- function( y, h, model=NULL ){
  f = msarima(y, h=h, model=model, initial="backcasting", 
              interval="none" )$forecast
  list(mean = f)
}

## ETS models
f_ets <- function( y, h, model=NULL ){
  fit = ets( y=y, model=model )
  forecast( fit, h=h )
}

## Holt winters
f_hw <- function( y, h, model=NULL ){
  fit = HoltWinters(y, alpha=model$alpha, beta=model$beta, 
                    gamma=model$gamma, seasonal = model$seasonal)
  list( mean = predict(fit, n.ahead=h) )
}

## Double Seasonal Holt Winters
f_dshw <- function( y, h, model=NULL ){
  dshw( y, h=h, model=model )
}

## TBATS
f_tbats <- function( y, h, model=NULL ){
  fit = tbats( y, model=model )
  forecast( fit, h=h )
}

## Neural Network
f_nnetar <- function(y, h, model=NULL){
  fit = nnetar( y, model=model )
  forecast( fit, h=h)
}


## LSTM
f_lstm = function( y, h=NULL, model=NULL, seas = c("W","DW","DWY"),
                   epochs=100, batch_size = 32, validation_split = 0.2){
  
  if(length(seas)>1){ seas = seas[1]; }
  colunas = NULL;
  if(seas == "D"){ colunas = 1:24; }
  if(seas == "W"){ colunas = 25:48; }
  if(seas == "DW"){ colunas = 1:48; }
  if(seas == "DWY"){ colunas = 1:72; }
  if(is.null(colunas)){ stop("seas not found"); }
  n_input = length(colunas)
  
  # Data normalization
  
  y_min = min(y)
  y_max = max(y)
  
  yy = (y - y_min) / (y_max - y_min)
  
  n = length(yy)
  
  # Preparation of data in temporal sequences considering up to three seasonal cycles (daily, weekly and annual)
  timesteps_daily <- 24         # daily
  timesteps_weekly <- 24 * 7    # weekly
  timesteps_yearly <- 24 * 365  # annual
  
  ## fit if model == null
  if( is.null(model) ){
    
    inicio = timesteps_yearly +24 
    
    X_train <- array(NA, dim = c(n - inicio, 72, 1) )
    y_train <- array(NA, dim = c(n - inicio, 1))
    
    for(i in 1:(n - inicio)){
      
      j = inicio+i ## index of yy
      
      y_train[i] = yy[j]
      
      X_train[i, , 1] <- c(
        yy[(j-1):(j-24)], ## last 24 hours
        yy[(j-timesteps_weekly-1):(j-timesteps_weekly-24)], ## 24 hours of one week ago
        yy[(j-timesteps_yearly-1):(j-timesteps_yearly-24)]  ## 24 hours of one year ago
      )
    }
    
    X_train = X_train[ , colunas, ,drop=F] ## select seasonal columns
    
    #Construction of the LSTM model
    model <- keras_model_sequential() 
    model %>%
      layer_lstm(units = 50, input_shape = c(n_input, 1)) %>%
      layer_dense(units = 1)
    
    # Model compilation
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adam()
    )
    
    # Model training
    history <- model %>% fit(
      x = X_train,
      y = y_train,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = validation_split
    )
    
  }## end if model
  
  
  ## forecast
  point_forecast = NULL
  
  if(!is.null(h)){
    
    X_test <- array(NA, dim = c(h, 72, 1))
    test_data = c( yy, rep(NA, h) )
    
    inicio = length(yy)
    
    for (i in 1:h) {
      
      j = inicio+i ## index of train_data
      
      X_test[i, , 1] <- c(
        test_data[(j-1):(j-24)], ## last 24 hours
        test_data[(j-timesteps_weekly-1):(j-timesteps_weekly-24)], ## 24 hours of one week ago
        test_data[(j-timesteps_yearly-1):(j-timesteps_yearly-24)]  ## 24 hours of one year ago
      )
      
      test_data[j] = predict(model, X_test[i,colunas,,drop=F], verbose=0)
      
    }
    
    test_data = test_data*(y_max - y_min) + y_min 
    
    point_forecast = tail(test_data, h)
  } 
  ##
  
  a = k_clear_session() ## attempt to reduce memory usage
  
  list( mean = point_forecast, model=model )  
  
}


