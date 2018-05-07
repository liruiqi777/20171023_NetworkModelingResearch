# READING IN ALL GENERATED DATA
# 100 agents with 10 initial adopters
path = "/Users/RLi/workspace/research/20171023_NetworkModelingResearch/initial_adopter_test_new_model/watts_only_rewire_factor_test/adopter_hist_run3/0326run/"
dataset_1_10_degree = read.table(paste(path, "watts_0.1_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_10_degree = read.table(paste(path, "watts_0.01_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_10_degree = read.table(paste(path, "watts_0.001_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_1_10_greedy = read.table(paste(path, "watts_0.1_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_10_greedy = read.table(paste(path, "watts_0.01_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_10_greedy = read.table(paste(path, "watts_0.001_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_1_10_influence = read.table(paste(path, "watts_0.1_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_10_influence = read.table(paste(path, "watts_0.01_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_10_influence = read.table(paste(path, "watts_0.001_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_10_degree = read.table(paste(path, "watts_0.2_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_10_degree = read.table(paste(path, "watts_0.02_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_10_degree = read.table(paste(path, "watts_0.002_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_10_greedy = read.table(paste(path, "watts_0.2_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_10_greedy = read.table(paste(path, "watts_0.02_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_10_greedy = read.table(paste(path, "watts_0.002_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_10_influence = read.table(paste(path, "watts_0.2_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_10_influence = read.table(paste(path, "watts_0.02_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_10_influence = read.table(paste(path, "watts_0.002_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_10_degree = read.table(paste(path, "watts_0.5_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_10_degree = read.table(paste(path, "watts_0.05_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_10_degree = read.table(paste(path, "watts_0.005_10_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_10_greedy = read.table(paste(path, "watts_0.5_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_10_greedy = read.table(paste(path, "watts_0.05_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_10_greedy = read.table(paste(path, "watts_0.005_10_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_10_influence = read.table(paste(path, "watts_0.5_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_10_influence = read.table(paste(path, "watts_0.05_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_10_influence = read.table(paste(path, "watts_0.005_10_influence_adopter_hist", sep=""), sep = " ", header = FALSE)


# reformatting
dataset_1_10_degree = colMeans(dataset_1_10_degree[,1:11])
dataset_01_10_degree = colMeans(dataset_01_10_degree[,1:11])
dataset_001_10_degree = colMeans(dataset_001_10_degree[,1:11])
dataset_1_10_greedy = colMeans(dataset_1_10_greedy[,1:11])
dataset_01_10_greedy = colMeans(dataset_01_10_greedy[,1:11])
dataset_001_10_greedy = colMeans(dataset_001_10_greedy[,1:11])
dataset_1_10_influence = colMeans(dataset_1_10_influence[,1:11])
dataset_01_10_influence = colMeans(dataset_01_10_influence[,1:11])
dataset_001_10_influence = colMeans(dataset_001_10_influence[,1:11])
dataset_2_10_degree = colMeans(dataset_2_10_degree[,1:11])
dataset_02_10_degree = colMeans(dataset_02_10_degree[,1:11])
dataset_002_10_degree = colMeans(dataset_002_10_degree[,1:11])
dataset_2_10_greedy = colMeans(dataset_2_10_greedy[,1:11])
dataset_02_10_greedy = colMeans(dataset_02_10_greedy[,1:11])
dataset_002_10_greedy = colMeans(dataset_002_10_greedy[,1:11])
dataset_2_10_influence = colMeans(dataset_2_10_influence[,1:11])
dataset_02_10_influence = colMeans(dataset_02_10_influence[,1:11])
dataset_002_10_influence = colMeans(dataset_002_10_influence[,1:11])
dataset_5_10_degree = colMeans(dataset_5_10_degree[,1:11])
dataset_05_10_degree = colMeans(dataset_05_10_degree[,1:11])
dataset_005_10_degree = colMeans(dataset_005_10_degree[,1:11])
dataset_5_10_greedy = colMeans(dataset_5_10_greedy[,1:11])
dataset_05_10_greedy = colMeans(dataset_05_10_greedy[,1:11])
dataset_005_10_greedy = colMeans(dataset_005_10_greedy[,1:11])
dataset_5_10_influence = colMeans(dataset_5_10_influence[,1:11])
dataset_05_10_influence = colMeans(dataset_05_10_influence[,1:11])
dataset_005_10_influence = colMeans(dataset_005_10_influence[,1:11])

dataset_1_5_degree = read.table(paste(path, "watts_0.1_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_5_degree = read.table(paste(path, "watts_0.01_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_5_degree = read.table(paste(path, "watts_0.001_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_1_5_greedy = read.table(paste(path, "watts_0.1_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_5_greedy = read.table(paste(path, "watts_0.01_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_5_greedy = read.table(paste(path, "watts_0.001_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_1_5_influence = read.table(paste(path, "watts_0.1_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_01_5_influence = read.table(paste(path, "watts_0.01_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_001_5_influence = read.table(paste(path, "watts_0.001_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_5_degree = read.table(paste(path, "watts_0.2_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_5_degree = read.table(paste(path, "watts_0.02_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_5_degree = read.table(paste(path, "watts_0.002_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_5_greedy = read.table(paste(path, "watts_0.2_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_5_greedy = read.table(paste(path, "watts_0.02_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_5_greedy = read.table(paste(path, "watts_0.002_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_2_5_influence = read.table(paste(path, "watts_0.2_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_02_5_influence = read.table(paste(path, "watts_0.02_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_002_5_influence = read.table(paste(path, "watts_0.002_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_5_degree = read.table(paste(path, "watts_0.5_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_5_degree = read.table(paste(path, "watts_0.05_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_5_degree = read.table(paste(path, "watts_0.005_5_degree_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_5_greedy = read.table(paste(path, "watts_0.5_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_5_greedy = read.table(paste(path, "watts_0.05_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_5_greedy = read.table(paste(path, "watts_0.005_5_greedy_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_5_5_influence = read.table(paste(path, "watts_0.5_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_05_5_influence = read.table(paste(path, "watts_0.05_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)
dataset_005_5_influence = read.table(paste(path, "watts_0.005_5_influence_adopter_hist", sep=""), sep = " ", header = FALSE)


# reformatting
dataset_1_5_degree = colMeans(dataset_1_5_degree[,1:11])
dataset_01_5_degree = colMeans(dataset_01_5_degree[,1:11])
dataset_001_5_degree = colMeans(dataset_001_5_degree[,1:11])
dataset_1_5_greedy = colMeans(dataset_1_5_greedy[,1:11])
dataset_01_5_greedy = colMeans(dataset_01_5_greedy[,1:11])
dataset_001_5_greedy = colMeans(dataset_001_5_greedy[,1:11])
dataset_1_5_influence = colMeans(dataset_1_5_influence[,1:11])
dataset_01_5_influence = colMeans(dataset_01_5_influence[,1:11])
dataset_001_5_influence = colMeans(dataset_001_5_influence[,1:11])
dataset_2_5_degree = colMeans(dataset_2_5_degree[,1:11])
dataset_02_5_degree = colMeans(dataset_02_5_degree[,1:11])
dataset_002_5_degree = colMeans(dataset_002_5_degree[,1:11])
dataset_2_5_greedy = colMeans(dataset_2_5_greedy[,1:11])
dataset_02_5_greedy = colMeans(dataset_02_5_greedy[,1:11])
dataset_002_5_greedy = colMeans(dataset_002_5_greedy[,1:11])
dataset_2_5_influence = colMeans(dataset_2_5_influence[,1:11])
dataset_02_5_influence = colMeans(dataset_02_5_influence[,1:11])
dataset_002_5_influence = colMeans(dataset_002_5_influence[,1:11])
dataset_5_5_degree = colMeans(dataset_5_5_degree[,1:11])
dataset_05_5_degree = colMeans(dataset_05_5_degree[,1:11])
dataset_005_5_degree = colMeans(dataset_005_5_degree[,1:11])
dataset_5_5_greedy = colMeans(dataset_5_5_greedy[,1:11])
dataset_05_5_greedy = colMeans(dataset_05_5_greedy[,1:11])
dataset_005_5_greedy = colMeans(dataset_005_5_greedy[,1:11])
dataset_5_5_influence = colMeans(dataset_5_5_influence[,1:11])
dataset_05_5_influence = colMeans(dataset_05_5_influence[,1:11])
dataset_005_5_influence = colMeans(dataset_005_5_influence[,1:11])


# AVERAGE BY INITIAL ADOPTER APPROACH AND NUMBER OF INITIAL ADOPTERS
initial_adopter_by_degree_5 = c(dataset_001_5_degree[1], dataset_002_5_degree[1], dataset_005_5_degree[1],
                                dataset_01_5_degree[1], dataset_02_5_degree[1], dataset_005_5_degree[1],
                                dataset_1_5_degree[1], dataset_2_5_degree[1], dataset_5_5_degree[1])
initial_adopter_by_greedy_5 = c(dataset_001_5_greedy[1], dataset_002_5_greedy[1], dataset_005_5_greedy[1],
                                dataset_01_5_greedy[1], dataset_02_5_greedy[1], dataset_05_5_greedy[1],
                                dataset_1_5_greedy[1], dataset_2_5_greedy[1], dataset_5_5_greedy[1])
initial_adopter_by_influence_5 = c(dataset_001_5_influence[1], dataset_002_5_influence[1], dataset_005_5_influence[1],
                                   dataset_01_5_influence[1], dataset_02_5_influence[1], dataset_05_5_influence[1],
                                   dataset_1_5_influence[1], dataset_2_5_influence[1], dataset_5_5_influence[1])

initial_adopter_by_degree_10 = c(dataset_001_10_degree[1], dataset_002_10_degree[1], dataset_005_10_degree[1],
                                 dataset_01_10_degree[1], dataset_02_10_degree[1], dataset_005_10_degree[1],
                                 dataset_1_10_degree[1], dataset_2_10_degree[1], dataset_5_10_degree[1])
initial_adopter_by_greedy_10 = c(dataset_001_10_greedy[1], dataset_002_10_greedy[1], dataset_005_10_greedy[1],
                                 dataset_01_10_greedy[1], dataset_02_10_greedy[1], dataset_05_10_greedy[1],
                                 dataset_1_10_greedy[1], dataset_2_10_greedy[1], dataset_5_10_greedy[1])
initial_adopter_by_influence_10 = c(dataset_001_10_influence[1], dataset_002_10_influence[1], dataset_005_10_influence[1],
                                    dataset_01_10_influence[1], dataset_02_10_influence[1], dataset_05_10_influence[1],
                                    dataset_1_10_influence[1], dataset_2_10_influence[1], dataset_5_10_influence[1])



plot_path = "/Users/RLi/workspace/research/20171023_NetworkModelingResearch/initial_adopter_test_new_model/watts_only_rewire_factor_test/adopter_hist_run3/0326run/"
pdf(paste(plot_path, "Plots.pdf", sep=""))

# PLOTTING ADOPTERS AT INITIAL EQUILIBRIUM WRT REWIRING FACTORS
g = 1.3*range(0, initial_adopter_by_degree_10,
          initial_adopter_by_greedy_10, initial_adopter_by_influence_10)

plot(initial_adopter_by_degree_10, type="b", ylim = g, col="blue", xaxt="n",
     main="Adopters at Initial Equilibrium With Respect to Rewiring Factor\n(10 initial adopters)")
lines(initial_adopter_by_greedy_10, type="b", col="red", pch = 3)
lines(initial_adopter_by_influence_10, type="b", col="green", pch = 2)

legend(g[2], text.width=c(0.3,2.0), c("by degree", "by greedy", "by influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)
axis(1, at=1:9, lab=c("0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"))


# PLOTTING ADOPTERS AT INITIAL EQUILIBRIUM WRT REWIRING FACTORS
g = 1.3*range(0, initial_adopter_by_degree_5,
              initial_adopter_by_greedy_5, initial_adopter_by_influence_5)

plot(initial_adopter_by_degree_5, type="b", ylim = g, col="blue", xaxt="n",
     main="Adopters at Initial Equilibrium With Respect to Rewiring Factor\n(5 initial adopters)")
lines(initial_adopter_by_greedy_5, type="b", col="red", pch = 3)
lines(initial_adopter_by_influence_5, type="b", col="green", pch = 2)

legend(g[2], text.width=c(0.3,2.0), c("by degree", "by greedy", "by influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)
axis(1, at=1:9, lab=c("0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"))



# AVERAGE ADOPTERS AFTER FREQUENT SHOCK WRT INITIAL APPROACH
g = 1.05 * range(dataset_001_10_degree, dataset_002_10_greedy, dataset_005_10_influence,
          dataset_01_10_degree, dataset_02_10_greedy, dataset_05_10_influence,
          dataset_1_10_degree, dataset_2_10_greedy, dataset_5_10_influence)
plot(dataset_001_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .001 and Initial Adopter = 10")
lines(dataset_001_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_001_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_002_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .002 and Initial Adopter = 10")
lines(dataset_002_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_002_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_005_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .005 and Initial Adopter = 10")
lines(dataset_005_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_005_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_01_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .01 and Initial Adopter = 10")
lines(dataset_01_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_01_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_02_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .02 and Initial Adopter = 10")
lines(dataset_02_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_05_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .05 and Initial Adopter = 10")
lines(dataset_05_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_05_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_1_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .1 and Initial Adopter = 10")
lines(dataset_1_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_1_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_2_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .2 and Initial Adopter = 10")
lines(dataset_2_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_2_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_5_10_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .5 and Initial Adopter = 10")
lines(dataset_5_10_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_5_10_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)



# AVERAGE ADOPTERS AFTER FREQUENT SHOCK WRT INITIAL APPROACH
g = 1.05 * range(dataset_001_5_degree, dataset_002_5_greedy, dataset_005_5_influence,
                 dataset_01_5_degree, dataset_02_5_greedy, dataset_05_5_influence,
                 dataset_1_5_degree, dataset_2_5_greedy, dataset_5_5_influence)
plot(dataset_001_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .001 and Initial Adopter = 5")
lines(dataset_001_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_001_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_002_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .002 and Initial Adopter = 5")
lines(dataset_002_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_002_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_005_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .005 and Initial Adopter = 5")
lines(dataset_005_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_005_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_01_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .01 and Initial Adopter = 5")
lines(dataset_01_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_01_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_02_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .02 and Initial Adopter = 5")
lines(dataset_02_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_05_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .05 and Initial Adopter = 5")
lines(dataset_05_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_05_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_1_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .1 and Initial Adopter = 5")
lines(dataset_1_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_1_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_2_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .2 and Initial Adopter = 5")
lines(dataset_2_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_2_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)

plot(dataset_5_5_degree[2:11], ylim = g, type="b", col="blue", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main = "Average Adopters After Frequent Shocks wrt Initial Adopter Selection
     Wiring Factor = .5 and Initial Adopter = 5")
lines(dataset_5_5_greedy[2:11], type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_5_5_influence[2:11], type="b", col="green", ann=FALSE, xaxt="n")
axis(1, at=1:10, lab=c("10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))
legend(g[2], text.width=c(0.5,1.5), c("degree", "greedy", "influence"),
       cex = 0.8, col=c("blue", "red", "green"), lty=1:3)


g = range(dataset_001_10_degree, dataset_002_10_degree, dataset_005_10_degree,
          dataset_01_10_degree, dataset_02_10_degree, dataset_05_10_degree,
          dataset_1_10_degree, dataset_2_10_degree, dataset_5_10_degree)
plot(dataset_001_10_degree, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by Degree Algorithm (n = 10)")
lines(dataset_002_10_degree, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_10_degree, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_10_degree, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_10_degree, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_10_degree, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_10_degree, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_10_degree, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_10_degree, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))


g = range(dataset_001_10_greedy, dataset_002_10_greedy, dataset_005_10_greedy,
          dataset_01_10_greedy, dataset_02_10_greedy, dataset_05_10_greedy,
          dataset_1_10_greedy, dataset_2_10_greedy, dataset_5_10_greedy)
plot(dataset_001_10_greedy, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by Greedy Algorithm (n = 10)")
lines(dataset_002_10_greedy, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_10_greedy, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_10_greedy, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_10_greedy, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_10_greedy, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_10_greedy, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_10_greedy, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_10_greedy, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))


g = range(dataset_001_10_influence, dataset_002_10_influence, dataset_005_10_influence,
          dataset_01_10_influence, dataset_02_10_influence, dataset_05_10_influence,
          dataset_1_10_influence, dataset_2_10_influence, dataset_5_10_influence)
plot(dataset_001_10_influence, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by Influence Algorithm (n = 10)")
lines(dataset_002_10_influence, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_10_influence, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_10_influence, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_10_influence, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_10_influence, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_10_influence, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_10_influence, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_10_influence, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))



g = range(dataset_001_5_degree, dataset_002_5_degree, dataset_005_5_degree,
          dataset_01_5_degree, dataset_02_5_degree, dataset_05_5_degree,
          dataset_1_5_degree, dataset_2_5_degree, dataset_5_5_degree)
plot(dataset_001_5_degree, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by degree Algorithm (n = 5)")
lines(dataset_002_5_degree, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_5_degree, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_5_degree, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_5_degree, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_5_degree, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_5_degree, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_5_degree, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_5_degree, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))

g = range(dataset_001_5_greedy, dataset_002_5_greedy, dataset_005_5_greedy,
          dataset_01_5_greedy, dataset_02_5_greedy, dataset_05_5_greedy,
          dataset_1_5_greedy, dataset_2_5_greedy, dataset_5_5_greedy)
plot(dataset_001_5_greedy, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by Greedy Algorithm (n = 5)")
lines(dataset_002_5_greedy, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_5_greedy, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_5_greedy, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_5_greedy, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_5_greedy, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_5_greedy, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_5_greedy, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_5_greedy, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))

g = range(dataset_001_5_influence, dataset_002_5_influence, dataset_005_5_influence,
          dataset_01_5_influence, dataset_02_5_influence, dataset_05_5_influence,
          dataset_1_5_influence, dataset_2_5_influence, dataset_5_5_influence)
plot(dataset_001_5_influence, ylim = g, type="b", col="grey90", xaxt="n",
     xlab = "rounds of shocks", ylab = "adopters at equilibrium",
     main="Average Adopters After Frequent Shocks wrt Rewiring Factors
     Initial Adopters Generated by influence Algorithm (n = 5)")
lines(dataset_002_5_influence, type="b", col="grey80", ann=FALSE, xaxt="n")
lines(dataset_005_5_influence, type="b", col="grey70", ann=FALSE, xaxt="n")
lines(dataset_01_5_influence, type="b", col="red", ann=FALSE, xaxt="n")
lines(dataset_02_5_influence, type="b", col="yellow", ann=FALSE, xaxt="n")
lines(dataset_05_5_influence, type="b", col="blue", ann=FALSE, xaxt="n")
lines(dataset_1_5_influence, type="b", col="grey30", ann=FALSE, xaxt="n")
lines(dataset_2_5_influence, type="b", col="grey20", ann=FALSE, xaxt="n")
lines(dataset_5_5_influence, type="b", col="grey10", ann=FALSE, xaxt="n")
axis(1, at=1:11, lab=c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"))


dev.off()
