import pandas as pd

PARAMETERS = ['bathrooms', 'sqft_living', 'sqft_lot', 'floors']
NUMBER = 4
LENGTH = 100

data = pd.read_csv('data.csv')

y = data['price'].to_numpy()
x = []
for i in PARAMETERS:
    x.append(data[i].to_numpy())

# weights
weights = [0 for i in range(NUMBER)]

# learning rate
alpha = 0.0000001

def calculateCost(weights):
    costs = []

    for i in range(LENGTH):
        cost = 0
        for j in range(NUMBER):
            cost += weights[j] * x[j][i]
        cost -= y[i]
        costs.append(cost)
    
    totalCost = sum(costs)/len(costs)

    return totalCost


def calculateSpecificCost(weights, index):
    cost = 0
    for i in range(NUMBER):
        cost += weights[i] * x[i][index]
    cost -= y[index]

    return cost


def gradientDescent(weights):
    adjustedWeights = [i for i in weights]
    gradient = [0 for _ in range(NUMBER)]

    for i in range(LENGTH):
        for j in range(NUMBER):
            gradient[j] += x[j][i] * (calculateSpecificCost(adjustedWeights, j))
    
    for i in range(NUMBER):
        adjustedWeights[i] -= alpha * gradient[i]
    
    return adjustedWeights


def train(epochs, epsilon, originalWeights):
    for i in range(1, epochs+1):
        newWeights = gradientDescent(originalWeights)
        cost = calculateCost(newWeights)
        print("Epoch:", str(i), " Loss:", str(cost))

        if abs(calculateCost(originalWeights) - cost) < epsilon:
            break
        originalWeights = newWeights

    print(originalWeights)


# TODO: create a function to test on the data not previously shown to the model and generate an accuracy percentage
def test(weights):
    raise NotImplementedError

train(100, 5.0, weights)
