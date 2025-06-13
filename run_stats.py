from stats import run_stats
from cart.cartAnn import run_experiment as AnnCartRE
from cart.cartSnn import run_experiment as SnnCartRE
from mountain_car.carAnn import run_experiment as AnnMountCarRE
from mountain_car.carSnn import run_experiment as SnnMountCarRE

#run_stats(10, 100, "cart/cartAnn_config.txt", "resultsAnnCart", AnnCartRE)
#run_stats(10, 100, "cart/cartSnn_config.txt", "resultsSnnCart", SnnCartRE)
run_stats(3, 100, "mountain_car/mountain_config_ann.txt", "resultsAnnMountainCar", AnnMountCarRE)
run_stats(3, 100, "mountain_car/mountain_config_snn.txt", "resultsSnnMountainCar", SnnMountCarRE)
