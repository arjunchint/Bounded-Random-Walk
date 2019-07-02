import numpy as np
import matplotlib.pyplot as plt

# Initialization of Random Seed for consistent experiments
# np.random.seed(21365)
True_W = [1/6,1/3,1/2,2/3,5/6]

# Create 100 Training Sets of 10 sequences each
TrainingSets=[]
for TrainingSet in range(100):
	sequences=[]
	for sequence in range(10):
		positions=[]
		pos = 2 
		positions.append(pos)
		while pos != 0 and pos != 4:
			pos += np.random.choice([-1, 1])
			positions.append(pos)
		trajectory = np.zeros((len(positions), 5))
		for counter,row in enumerate(trajectory):
			row[positions[counter]]=1
		sequences.append(trajectory)
	TrainingSets.append(sequences)



Lambdas=[0,.3,.8,1]
rel_error_by_lambda = []
for Lambda in Lambdas:
	alphas=np.linspace(0,.7,15)
	rel_error_by_alpha = []
	for alpha in alphas:
		RMSEs=[]
		for TrainingSet in TrainingSets:
			# Initialize weights
			W = np.array([[.5,.5,.5,.5,.5]])
			
			for X in TrainingSet:
				delta_w = np.zeros((1,5))
				for t in range(len(X)):
					P_k=0
					for k in range(1,t+2):
						P_k += Lambda**(t+1-k)*X[k-1]
					if X[t,0] == 1:
						delta_w += alpha*(np.zeros((1,X.shape[1]))-np.dot(W,X[t]))*P_k
						break
					elif X[t,-1]==1:
						delta_w += alpha*(np.ones((1,X.shape[1]))-np.dot(W,X[t]))*P_k
						break		
					delta_w += alpha*(np.dot(W,X[t+1])-np.dot(W,X[t]))*P_k
				W += delta_w
			RMSE = np.sqrt(((W - True_W) ** 2).mean()) 	
			RMSEs.append(RMSE)
		print("FINAL RMSE @ ", Lambda,alpha," : ",np.mean(RMSEs))
		rel_error_by_alpha.append(np.mean(RMSEs))
	# Plotting subsection of results to be able to compare to graph
	plot_alphas,plot_errors=[],[]
	for counter,err in enumerate(rel_error_by_alpha):
		if err < .7:
			plot_errors.append(err)
			plot_alphas.append(alphas[counter])

	plt.plot(plot_alphas,plot_errors,label=str(Lambda))
plt.ylim(ymax=.8) 
plt.xlabel('Alpha', fontsize=18)
plt.ylabel('RMSE Error', fontsize=16)
plt.legend()
plt.show()