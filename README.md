# VehiclesOnVideoDetection
This program reads local video (could use youtube video or live stream, but it has issues while playing with frame losses) with defined interval in seconds and counts vehicles on image. Then this numbers save into csv file for further processing.
We used OLS method for lineal aproximation for data values in 1 hour intervals, then square approximation for 12 in-hour values to get dependence of the average number of vehicles by time
