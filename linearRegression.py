from numpy import *

def computer_error_for_given_points(b, m, points):
    #initialize error
    total_error = 0
    #for every point
    for i in range(0, len(points)):
        #get the x value
        x = points[i,0]
        #get the y value
        y = points[i,1]
        #get the difference, square it and add it to the total
        total_error += (y - (m*x +b))**2
    
    #get the average
    return total_error/float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, numOfIterations, learningRates):
    #starting b and m
    b = starting_b
    m = starting_m

    #gradient Descent
    for i in range(numOfIterations):
        #update b and m with the new more accurate b and m by performing this gradient stop

        b,m = step_gradient(b,m, array(points), learningRates)

    return [b,m]

def step_gradient(b_current, m_current, points, learningRates):

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        #direction with respect to b and m
        #computing partial derivatives of our error function

        b_gradient += -(2/N) * (y-((m_current*x)+b_current))
        m_gradient += -(2/N) * x * (y - ((m_current*x)+b_current))
    
    #update our b and m values using this partial derivative
    new_b = b_current- (learningRates * b_gradient)
    new_m = m_current - (learningRates * m_gradient)

    return [new_b,new_m]

def run():
    #Step - 1 - Collect Data
    points = genfromtxt('data.csv', delimiter=',')

    #Step - 2 - define our hyperparameters
    learningRates = 0.0001
    #y = mx + b (slope formula)
    initialB = 0
    initialM = 0
    numOfIterations = 1000

    #Step - 3 - Train our model

    print("Starting gradient descent at b = {0}, m={1}, error={2}".format(initialB, initialM, computer_error_for_given_points(initialB, initialM, points)))

    [b,m] = gradient_descent_runner(points, initialB, initialM,numOfIterations, learningRates)

    print("Ending gradient descent at b = {1}, m={2}, error={3}".format(numOfIterations,b,m,computer_error_for_given_points(b, m, points)))


if __name__ == '__main__':
    run()