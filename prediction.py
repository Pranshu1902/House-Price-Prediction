import pandas as pd

PARAMETERS = ['bathrooms', 'sqft_living', 'sqft_lot', 'floors']
NUMBER = 4
LENGTH = 100
TEST = 50

data = pd.read_csv('data.csv')

y = data['price'].to_numpy()
x = []
for i in PARAMETERS:
    x.append(data[i].to_numpy())

# weights
weights = []
for i in range(NUMBER):
    weights.append(0.0)

# learning rate
alpha = 0.0000001

def calculateCost(weights):
    costs = []

    for i in range(LENGTH):
        cost = 0
        for j in range(NUMBER):
            cost += weights[j] * x[j][i]
        cost -= y[i]
        cost = cost**2
        costs.append(cost)

    return sum(costs)


def calculateSpecificCost(weights, index):
    cost = 0
    for i in range(NUMBER):
        cost += weights[i] * x[i][index]
    cost -= y[index]

    return cost


def gradientDescent(weights):
    adjustedWeights = []
    for i in weights:
        adjustedWeights.append(i)

    gradient = [0 for _ in range(NUMBER)]

    for i in range(LENGTH):
        for j in range(NUMBER):
            gradient[j] += -2 * x[j][i] * calculateCost(adjustedWeights)
    
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


def predict(weights, input):
    prediction = 0
    for i in range(NUMBER):
        prediction += weights[i] * x[i][input]
    return prediction

print(x)
print(len(x))

def test(weights):
    scores = []

    for i in range(LENGTH+1, LENGTH+TEST+1):
        prediction = predict(weights, i)
        # calculate accuracy
        error = 100.0*(y[i] - abs(prediction - y[i]))/y[i]
        scores.append(error)
        print(error)

    print("Accuracy:", sum(scores)/len(scores))


EPOCHS = 300

train(EPOCHS, 5.0, weights)
test(weights)
