from stats import run_stats
from cart.cartAnn import run_experiment as AnnCartRE
from cart.cartSnn import run_experiment as SnnCartRE
from mountain_car.carAnn import run_experiment as AnnMountCarRE
from mountain_car.carSnn import run_experiment as SnnMountCarRE
from pendulum.pendulumAnn import run_experiment as AnnPendulumRE
from pendulum.pendulumSnn import run_experiment as SnnPendulumRE

#run_stats(30, 100, "cart/cartAnn_config.txt", "resultsAnnCart", AnnCartRE)
#run_stats(30, 100, "cart/cartSnn_config.txt", "resultsSnnCart", SnnCartRE)
#run_stats(15, 100, "mountain_car/mountain_config_ann.txt", "resultsAnnMountainCar", AnnMountCarRE)
#run_stats(15, 100, "mountain_car/mountain_config_snn.txt", "resultsSnnMountainCar", SnnMountCarRE)
run_stats(30, 100, "pendulum/pendulum_config_ann.txt", "resultsAnnPendulum", AnnPendulumRE)
run_stats(30, 100, "pendulum/pendulum_config_snn.txt", "resultsSnnPendulum", SnnPendulumRE)