![Environments running w train](https://github.com/alexandre2004costa/Spinking-Neural-Networks/blob/master/SNN.gif)

# 🧠 Spiking Neural Networks (SNN)

This project explores the application of **Spiking Neural Networks (SNNs)** in classic control environments such as CartPole, MountainCar, Pendulum, and others. It uses a firing-rate-based model (`Rate Izhikevich`) combined with the **NEAT** algorithm to evolve network topology.

## 📚 About the Project

SNNs are a more biologically plausible model of neural networks, where neurons communicate via spikes. This project implements a simplified rate-based version and compares its performance with traditional artificial neural networks (ANNs).


## 🧪 Tested Environments

- `CartPole`
- `CartPole Continuous`
- `MountainCar`
- `Pendulum`
- `LunarLander` 

## ⚙️ Project Structure
```
├── cartPole.py
├── rate_iznn.py
├── rate_iznn_cont.py
├── run_stats.py
├── stats.py
├── structure.txt
├── cart
│ ├── cartAnn.py
│ ├── cartAnn_config.txt
│ ├── cartSnn.py
│ ├── cartSnn_config.txt
│ ├── customIzGenome.py
│ └── init.py
├── cartCont
│ ├── cartAnn.py
│ ├── cartAnn_config.txt
│ ├── cartSnn.py
│ ├── cartSnn_config.txt
│ ├── customIzGenome.py
│ └── init.py
├── lunar
│ ├── customIzGenome.py
│ ├── lunarAnn.py
│ ├── lunarSnn.py
│ ├── lunar_config_ann.txt
│ ├── lunar_config_snn.txt
│ ├── neat_iznn.py
│ └── rate_iznn.py
├── mountain_car
│ ├── carAnn.py
│ ├── carSnn.py
│ ├── customIzGenome.py
│ ├── mountain_config_ann.txt
│ ├── mountain_config_snn.txt
│ └── init.py
├── pendulum
│ ├── customIzGenome.py
│ ├── pendulumAnn.py
│ ├── pendulumSnn.py
│ ├── pendulum_config_ann.txt
│ └── pendulum_config_snn.txt
└── results
├── resultsAnnCart30.csv
├── resultsAnnCartCont30.csv
├── resultsAnnMountainCar30.csv
├── resultsAnnPendulum30.csv
├── resultsSnnCart30.csv
├── resultsSnnCartCont30.csv
├── resultsSnnMountainCar30.csv
├── resultsSnnPendulum30.csv
└── Snn Results.xlsx
```

## 🚀 Running the Code

```bash
# Example: run CartPole with SNN
python cart/cartSnn
