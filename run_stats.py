from stats import run_stats
from cart.cartAnn import run_experiment as Ann_run_experiment
from cart.cartSnn import run_experiment as Snn_run_experiment

run_stats(10, 100, "cart/cartAnn_config.txt", "resultsAnnCart", Ann_run_experiment)
run_stats(10, 100, "cart/cartSnn_config.txt", "resultsSnnCart", Snn_run_experiment)