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



Lambdas=[0,.1,.3,.5,.7,.9,1]
rel_error_by_lambda = []

# Iterate through Lambdas
for Lambda in Lambdas:
	alphas=[.005]
	rel_error_by_alpha = []
	for alpha in alphas:
		RMSEs=[]
		for TrainingSet in TrainingSets:
			rel_error = 1
			RMSE = 10
			# Initialize weights
			W = np.zeros((1,5))
			iterat=0
			# Repeatedly presenting training set till convergence
			while rel_error > .0001:
				RMSE1 = RMSE
				W1=W			
				delta_w = np.zeros((1,5))
				for X in TrainingSet:
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
				rel_error = max(abs(delta_w[0]))
				iterat+=1
			# print("Iterations: ", iterat)
			RMSE = np.sqrt(((W - True_W) ** 2).mean()) 	
			RMSEs.append(RMSE)
		print("FINAL RMSE @ ", Lambda,alpha," : ",np.mean(RMSEs))
		rel_error_by_alpha.append(np.mean(RMSEs))
	rel_error_by_lambda.append(min(rel_error_by_alpha))
plt.plot(Lambdas,rel_error_by_lambda)

plt.xlabel('Lambda', fontsize=18)
plt.ylabel('RMSE Error', fontsize=16)
plt.show()