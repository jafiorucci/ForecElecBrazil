source('Libraries.R')

#---------------- Descritive analisys-----------------------------------------

## data
df <- read.csv("PreProcessedData.csv")
df$datetime <- as.POSIXct(df$datetime , format = "%Y-%m-%d %H:%M:%S") 
x <- xts(df[,-1], order.by = df$datetime)

x_ <- msts( x, seasonal.periods=c(24, 24*7, 365.25*24) )
out = x_ %>% mstl()
trend = out[,"Trend"]

x_trend_adj = x - as.numeric(trend) ## trend adjusted
x_trend_adj = x_trend_adj + mean( as.numeric(trend) )

rm(x_); rm(out); rm(trend);

rm(df)
##


## ---- Fig complete time series ----
png("serie_anual.png", width = 600, height = 400)

plot.xts(x, main="", ylim=c(20000, 55000), format.labels="%Y")

dev.off()


## ---- Fig few weeks of time series ----
#png("serie_zoom.png", width = 600, height = 400)
pdf( file = "serie_zoom.pdf", width=8, height = 4)

plot.xts(x["20181001::20181021"], main="", ylim=c(23000, 50000), 
         format.labels="%a", las = 2)

dev.off()


## ---- Fig boxplots by months ----
library(lubridate)
library(ggplot2)
library(dplyr)

Sys.setlocale("LC_TIME", "English")

df_box <- data.frame(
  date = index( x_trend_adj ),
  value = coredata( x_trend_adj ),
  month = months( index( x_trend_adj ) )
)

df_box$month <- factor(
  df_box$month,
  levels = month.name
)

pdf( file = "boxplot_month.pdf", width=7, height = 4)

ggplot(df_box, aes(x = month, y = value)) +
  geom_boxplot(fill = "lightblue") +
  labs(
    title = "Boxplot of Hourly Electricity Demand by Month",
    x = "Month", y = "Demand (MWh/h)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

dev.off()

rm(df_box);


## ---- Fig spectrum density ----

x_ts <- ts(coredata(x_trend_adj), frequency = 1)

pdf("spectrum.pdf", width = 9, height = 5)

# Divide a área gráfica em 2 linhas e 1 coluna
par(mfrow = c(2,1), mar = c(4,4,3,2))  # margens para melhor visualização

# (a) Espectro para frequências baixas (anual)
spectrum(x_ts, log = "no", main = "(a) Spectral Density", xlim = c(0, 0.0012))

annual_col <- rgb(0, 0.5, 0, 0.2)  # verde semi-transparente
abline(v = 1/8766, col = annual_col, lwd = 8, lty = 1)

legend("top", legend = "Annual cycle (1/8766)", col = annual_col, lwd = 8, lty = 1, bty = "n")

# (b) Espectro para frequências altas (diário e semanal)
spectrum(x_ts, log = "no", main = "(b) Spectral Density", xlim = c(0, 0.05))

daily_col <- rgb(1, 0, 0, 0.2)   # vermelho semi-transparente
weekly_col <- rgb(0, 0, 1, 0.2)  # azul semi-transparente

abline(v = 1/24, col = daily_col, lwd = 8, lty = 1)
abline(v = 1/168, col = weekly_col, lwd = 8, lty = 1)

legend("top", legend = c("Daily cycle (1/24)", "Weekly cycle (1/168)"),
       col = c(daily_col, weekly_col), lwd = 8, lty = 1, bty = "n")

dev.off()

rm(x)
rm(x_ts)
rm(annual_col)
rm(daily_col)
rm(weekly_col)
rm(x_trend_adj)



