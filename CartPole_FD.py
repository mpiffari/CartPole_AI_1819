# ====== IMPORT ================
import math
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import datetime


# ====== SUPPORT FUNCTIONS ================
def cart_rollout(initialState, w):
    next_state = initialState
    time = 0
    reward = 0

    while time < thresholdTime and abs(next_state[1]) < thresholdAngle:
        # Use the linear RBF network to approximate learn
        # action = pi(s,W) = W' * phi(s,a)
        phi_value = phi(next_state)
        u = np.dot(w.transpose(), phi_value)
        step_state = cart_dynamics(next_state, u)
        step_state *= sampleTime
        next_state = next_state + step_state
        if abs(next_state[1]) < thresholdAngle:
            reward = reward + sampleTime
        time = time + sampleTime
    return reward


# Definition of the basis functions
def phi(s):
    # Discretization only along angular and angular velocity, son we're interested only
    # on this two centrum
    s = np.array([s[1], s[3]])

    matrix = matlib.repmat(s.transpose(), numberOfCentrum, 1)
    matrix_2 = matlib.repmat(iS.transpose(), numberOfCentrum, 1)
    matrix = Centrum - matrix
    matrix **= 2
    matrix *= matrix_2
    result = np.exp(- 0.5 * np.sum(matrix, 1))
    result = np.reshape(result, (len(result), 1))  # Return always a column vector
    return result


# Definition of the dynamics of the cart using the following matrix
def cart_dynamics(X, u):
    # State: X = [x; th; v; w]'

    linear_position = X[0]
    angle = X[1]
    linear_velocity = X[2]
    angular_velocity = X[3]

    q = np.array([linear_position,
                  angle])
    dq = np.array([linear_velocity,
                   angular_velocity])

    # Matrix for dynamic definition
    H_dynamic = np.array([[poleMass + cartMass, - poleMass * poleLength * math.cos(q[1])],
                  [- poleMass * math.cos(q[1]), poleMass * poleLength]])
    C_dynamic = np.array([[0, poleMass * poleLength * math.sin(q[1] * dq[1])],
                          [0, 0]])
    G_dynamic = np.array([[0],
                          [- poleMass * poleLength * gravity * math.sin(q[1])]])
    B_dynamic = np.array([[1],
                          [0]])

    ddq = np.dot(np.linalg.inv(H_dynamic), (B_dynamic.dot(u) - C_dynamic.dot(dq) - G_dynamic))
    dx = np.concatenate((dq, ddq), axis=0)
    return dx


# ====== CART DEFS ================
# Parameters of the cart
cartMass = 1.0
# Parameters of the pole
poleMass = 0.5
poleLength = 0.75
# Other parameters
gravity = 9.81  # Gravity
sampleTime = 0.05
iterMax = 2000  # Maximum number of iterations of the algorithm
thresholdTime = 200
trialNumber = 1
thresholdAngle = np.pi / 6
thresholdOmega = np.sqrt((1 - np.cos(thresholdAngle)) * gravity / poleLength)
# Flag activation
debug = True
plot = True

# ====== PARTITION OF THE INPUT SPACE FOR OVERLAPPING WITH GAUSSIAN NEURONS (RBF)  ================
# Definition of the RBF policy: center and variance computation
discrete_angle = 5  # Units for the angular variable
discrete_angular_velocity = 5  # Units for the angular velocity variable
number_of_centrum = discrete_angle * discrete_angular_velocity
mu = np.zeros((number_of_centrum, 2))
sigma = np.zeros((number_of_centrum, 2))
row = 0

# MANUAL LINSPACE
# angle_step = (2 * zMax) / (discrete_angle - 1)
# angularVelocity_step = (wMax * 2) / (discrete_angular_velocity - 1)
# sigma_position = angle_step / np.sqrt(2 * number_of_centrum)
# sigma_velocity = angularVelocity_step / np.sqrt(2 * number_of_centrum)
#
# # [sigma_angle, sigma_angularVelocity]
# for i in range(-zMax, zMax, angle_step):
#     for j in range(-wMax, wMax, angularVelocity_step):
#         mu[[row], :] = np.array([i, j])
#         row = row + 1
#
# for i in range(1, number_of_centrum):
#     sigma[[i], :] = np.array([sigma_position, sigma_velocity])

# AUTOMATIC LINSPACE

Z = np.linspace(-thresholdAngle, thresholdAngle, num=discrete_angle)  # Angle variable
dZ = (thresholdAngle - (-thresholdAngle)) / (discrete_angle - 1)
W = np.linspace(-thresholdOmega, thresholdOmega, num=discrete_angular_velocity)  # Angular velocity
dW = (thresholdOmega - (-thresholdOmega)) / (discrete_angular_velocity - 1)
iS = np.array([[3 / np.sqrt(dZ)], [3 / np.sqrt(dW)]])  # Width of the kernel
# Compute the centers of the feature vectors
Centrum = np.zeros((number_of_centrum, 2))

# Creation of the centrum map (sequence of points described by coordinates)
for xii in range(0, discrete_angle):
    for vii in range(0, discrete_angular_velocity):
        Centrum[[row], :] = np.array([Z[xii], W[vii]])
        row = row + 1
numberOfCentrum = len(Centrum)

if plot:
    colors = (0, 0, 0)
    area = np.pi * 3
    plt.scatter(Centrum[:, [0]], Centrum[:, [1]], s=area, c=colors, alpha=0.5)
    plt.title('Centrums of the RBF network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# ====== PARAMETERS OF THE RL ALGORITHM ================
n_iteration_stable = 0 # Parameter for early stop of the iteration
rolloutMax = 2.5 * numberOfCentrum  # No. of rollouts twice the size of the parameter vector
lambda_l = 0.0  # Regularization factor (zero since there are enough samples)
variance_of_perturbation = 0.1  # Variance of the random Gaussian perturbation of the parameters
alpha = 0.4  # Learning rate

# Declaration of variables to store the learning results
allTrialRewards = []  # Store all the returns for all the trials
trialRewards = []  # Store the return for the current trial
trialTimeDuration = [] # Store the return execution time for each trial

# ====== ALGORITHM EXECUTION ================
main_start_time = datetime.datetime.now()
for trial in range(0, trialNumber):
    if not debug:
        # Minimal debug
        print("Trial " + str(trial))
    trial_start_time = datetime.datetime.now()
    np.random.seed(trial)  # Set the random seed

    # Initialization
    W = np.zeros((numberOfCentrum, 1))  # Weights
    J = 0  # Initialise cost to zero
    trialRewards = []  # Initialise trial return to empty

    # Test without any training
    # Compute return before training for 'rolloutMax' random rollouts
    for rollout in range(0, numberOfCentrum):
        random_initial_state = np.array([[0], [2 * thresholdAngle * (np.random.rand() - 0.5)], [0], [0]])  # Random initial state
        J = J + cart_rollout(random_initial_state, W)

    J = J / numberOfCentrum
    trialRewards = np.array([J])  # Store the average cost for untrained RBF
    tGN = np.array(())
    # Print information for untrained network
    if debug:
        print('NOT TRAINED Trial: ' + str(trial) + ' Iteration: 0 AverRerturn ' + str(J))

    # Main loop of the algorithm
    for iteration in range(1, iterMax):
        if not debug:
            # Minimal debug
            print("         Iteration " + str(iteration))
        if debug:
            print('Trial: ' + str(trial) + ' Iteration: ' + str(iteration), end='')  # print iterarion no.

        dJ = []  # Set the cost change for perturbations to empty
        Delta = []  # Set the perturbation to empty

        # Simulate the system, rollouts
        for rollout in range(0, int(round(rolloutMax, 0))):
            # Inizializza lo stato iniziale s0
            random_initial_state = np.array([[0], [2 * thresholdAngle * (np.random.rand() - 0.5)], [0], [0]])  # random initial state

            # Genera una variazione randomica
            delta = variance_of_perturbation * np.random.randn(numberOfCentrum, 1)  # random parameter variation (Gaussian)

            # Perturbation
            J_positive = cart_rollout(random_initial_state, W + delta)  # Positive change return
            J_negative = cart_rollout(random_initial_state, W - delta)  # Negative change return

            difference = np.array([J_positive - J_negative])
            dJ = np.concatenate((dJ, difference), axis=0)  # Add return variation to dJ
            if len(Delta) == 0:
                Delta = np.transpose(delta)
            else:
                Delta = np.concatenate((Delta, np.transpose(delta)), axis=0)   # Add random perturbation on the parameters to Delta

        # Estimate the gradient from dJ and Delta (least squares)
        dJ = np.reshape(dJ, (len(dJ), 1))
        G_1 = np.linalg.inv(np.dot(np.transpose(Delta), Delta) + (lambda_l * np.eye(numberOfCentrum)))
        G_2 = np.dot(G_1, np.transpose(Delta))
        G = np.dot(G_2, dJ)
        G /= 2
        # Weight update
        G *= alpha
        W = W + G

        grad = np.linalg.norm(G, ord=2)
        if debug:
            print(' GradNorm: ' + str(grad), end='')  # Print gradient norm (debug)

        if grad == 0:
            n_iteration_stable += 1
            if n_iteration_stable == 3:
                print("EARLY INTERRUPTION: minimum obtained earlier")
                break

        if len(tGN) == 0:
            tGN = np.array([np.linalg.norm(G)])
        else:
            tGN = np.concatenate((tGN, np.array([np.linalg.norm(G)])), axis=0)

        # Evaluate the updated policy (This step is separated from the
        # training because we use the central differences method)
        J = 0  # Initialize cost to zero
        # Compute return at this stagefor 'rolloutMax' random rollouts
        for rollout in range(1, numberOfCentrum):
            random_initial_state = np.array([[0], [2 * thresholdAngle * (np.random.rand() - 0.5)], [0], [0]])  # Random initial state
            J = J + cart_rollout(random_initial_state, W)

        J = J / numberOfCentrum  # Average cost
        J_array = np.array([J])
        if len(trialRewards) == 0:
            trialRewards = J_array  # Store the average cost for the RBF
        else:
            trialRewards = np.concatenate((trialRewards, J_array), axis=0)  # Store the average cost for the RBF

        if debug:
            print(' AverReturn: ' + str(J))

    trialRewards = np.reshape(trialRewards, (len(trialRewards), 1))
    if len(allTrialRewards) == 0:
        allTrialRewards = trialRewards
    else:
        allTrialRewards = np.concatenate((allTrialRewards, trialRewards), axis=1)

    trial_end_time = datetime.datetime.now()
    trial_duration = (trial_end_time - trial_start_time).total_seconds()
    if len(trialTimeDuration) == 0:
        trialTimeDuration = np.array([trial_duration])
    else:
        trialTimeDuration = np.concatenate((trialTimeDuration, np.array([trial_duration])), axis=0)

main_end_time = datetime.datetime.now()
main_time_duration = (main_end_time - main_start_time).total_seconds()

if plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.text(0.95, 0.01, 'Execution time ' + str(main_time_duration) +' s',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)

    plt.title('Trend of reward')
    plt.ylabel('Reward')
    plt.xlabel('Episodes')

    number = trialNumber
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for i in range(len(allTrialRewards[0])):
        color = colors[i]
        plt.plot([x for x in range(len(allTrialRewards))], allTrialRewards[:, [i]], color=color, label=str(i+1))

plt.legend()
plt.show()
