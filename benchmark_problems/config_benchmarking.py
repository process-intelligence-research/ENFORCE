# Hard Constraints settings
# (select "MLP" and config soft constraints below for soft constraints method, select "BOTH" to run both models sequentially in a single run)
MODEL = "BOTH"  # Model type, options: "ENFORCE", "MLP", "BOTH"

# Soft Constraints settings
SOFT_CONSTRAINED = False
WEIGHT_LOSS_SOFT = 0

# Problem settings
PROBLEM = "function_fitting"  # Problem type, options: "function_fitting", "nonconvex_linear", "nonconvex_nonlinear", "extraction_column"

################################################################################
# Do not change anything below if you want to reproduce the paper's results #
################################################################################

# General settings
PLOT = True
SAVE = True
FIX_SEED = False  # Fix seed for reproducibility (among different runs)
N = 5  # Number of runs
VERBOSE = False

# Backpropagation through projection
# True  → Implicit Function Theorem backward
# False → unrolled autograd
IFT_BACKWARD = False

# Parametric optimization benchmark settings (only relevant if PROBLEM is "nonconvex_linear" or "nonconvex_nonlinear")
N_VARIABLES_OPT = 100  # Number of variables (optimization variables)
N_CONSTRAINTS_OPT = 50  # Number of constraints (optimization constraints)

# Hard Constraints settings
if PROBLEM in ("function_fitting", "extraction_column"):
    WEIGHT_LOSS_DISPLACEMENT = 0.5
    EPOCH_START_HARD_CONSTRAINED = 0
    TRAINING_TOLERANCE = 0.0001
    INFERENCE_TOLERANCE = 1e-6
    MAX_IT = 100
    PROJ_WEIGHTING_OPTION = 5
    ADA_NP_AUTO_ACTIVATION = True
elif PROBLEM == "pooling":
    WEIGHT_LOSS_DISPLACEMENT = 0.5
    EPOCH_START_HARD_CONSTRAINED = 0
    TRAINING_TOLERANCE = 1e-4
    INFERENCE_TOLERANCE = 1e-4
    MAX_IT = 100
    PROJ_WEIGHTING_OPTION = 1
    ADA_NP_AUTO_ACTIVATION = True
    POOLING_EPS_CHOL = 1e-3  # Cholesky regularisation
    REGULARISE_GRAM = True
elif PROBLEM == "sin_ineq":
    WEIGHT_LOSS_DISPLACEMENT = 0.5
    EPOCH_START_HARD_CONSTRAINED = 0
    TRAINING_TOLERANCE = 1e-4
    INFERENCE_TOLERANCE = 1e-4
    MAX_IT = 100
    PROJ_WEIGHTING_OPTION = 1
    ADA_NP_AUTO_ACTIVATION = False  # always run ada_np regardless of loss improvement
elif PROBLEM == "nonconvex_linear" or PROBLEM == "nonconvex_nonlinear":
    WEIGHT_LOSS_DISPLACEMENT = 0.5
    EPOCH_START_HARD_CONSTRAINED = 0
    TRAINING_TOLERANCE = 1e-3
    INFERENCE_TOLERANCE = 1e-3
    MAX_IT = 100
    PROJ_WEIGHTING_OPTION = 1
    ADA_NP_AUTO_ACTIVATION = True

if PROBLEM != "pooling":
    REGULARISE_GRAM = False

# NN settings
if PROBLEM == "function_fitting":
    INPUT_NEURONS = 1
    HIDDEN_NEURONS = 64
    OUTPUT_NEURONS = 2
    HIDDEN_LAYERS = 1
    # Learning mode
    SUPERVISED = True

elif PROBLEM == "extraction_column":
    INPUT_NEURONS = 3
    HIDDEN_NEURONS = 64
    OUTPUT_NEURONS = 9
    HIDDEN_LAYERS = 1
    # Learning mode
    SUPERVISED = True

elif PROBLEM == "pooling":
    INPUT_NEURONS = 4
    HIDDEN_NEURONS = 64
    OUTPUT_NEURONS = 5
    HIDDEN_LAYERS = 1
    # Learning mode
    SUPERVISED = True

elif PROBLEM == "sin_ineq":
    INPUT_NEURONS = 1
    HIDDEN_NEURONS = 64
    OUTPUT_NEURONS = 1 
    HIDDEN_LAYERS = 1
    SUPERVISED = True

elif PROBLEM == "nonconvex_linear" or PROBLEM == "nonconvex_nonlinear":
    INPUT_NEURONS = N_CONSTRAINTS_OPT
    HIDDEN_NEURONS = 200
    OUTPUT_NEURONS = N_VARIABLES_OPT
    HIDDEN_LAYERS = 2
    # Learning mode
    SUPERVISED = False

if PROBLEM == "function_fitting":
    # Training settings
    EPOCHS = 50000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1000
elif PROBLEM == "extraction_column":
    # Training settings
    EPOCHS = 1200
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
elif PROBLEM == "pooling":
    # Training settings
    EPOCHS = 10000
    LEARNING_RATE = 0.5e-4
    BATCH_SIZE = 64
elif PROBLEM == "sin_ineq":
    EPOCHS = 500
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    SIN_INEQ_N_TRAIN = 1200
    SIN_INEQ_N_TEST = 300
    SIN_INEQ_LEFT = 0.0
    SIN_INEQ_RIGHT = 9.4248     # ~3*pi
    SIN_INEQ_NOISE_STD = 0.3   # symmetric Gaussian noise std
    SIN_INEQ_NOISE_BIAS = 0.1  # asymmetric bias: negative at small x, positive at large x
elif PROBLEM == "nonconvex_linear" or PROBLEM == "nonconvex_nonlinear":
    EPOCHS = 1000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 200

# Data generation settings (only relevant for function fitting / inequality fitting)
DATA_TRAINING = 1000
DATA_TEST = 100000
LEFT_LIMIT = -2.0
RIGHT_LIMIT = 2.0

# Optimization problem data
if PROBLEM == "nonconvex_linear" or PROBLEM == "nonconvex_nonlinear":
    INPUT_DATA_PATH = f"data/raw/{PROBLEM}/X_data_random_nonlinearnonconvex_dataset_dim{N_VARIABLES_OPT}_eq{N_CONSTRAINTS_OPT}_ex10000_int-5_5.csv"
    OUTPUT_DATA_PATH = f"data/raw/{PROBLEM}/Y_data_random_nonlinearnonconvex_dataset_dim{N_VARIABLES_OPT}_eq{N_CONSTRAINTS_OPT}_ex10000_int-5_5.csv"
PARAMS_PATH = f"data/raw/{PROBLEM}/parameters_random_nonlinearnonconvex_dataset_dim{N_VARIABLES_OPT}_eq{N_CONSTRAINTS_OPT}_ex10000_int-5_5.pkl"
if PROBLEM == "extraction_column":
    INPUT_DATA_PATH_TRAIN = f"data/raw/{PROBLEM}/ED_Col_Data_x_train.csv"
    OUTPUT_DATA_PATH_TRAIN = f"data/raw/{PROBLEM}/ED_Col_Data_y_train.csv"
    INPUT_DATA_PATH_VAL = f"data/raw/{PROBLEM}/ED_Col_Data_x_val.csv"
    OUTPUT_DATA_PATH_VAL = f"data/raw/{PROBLEM}/ED_Col_Data_y_val.csv"
    INPUT_DATA_PATH_TEST = f"data/raw/{PROBLEM}/ED_Col_Data_x_test.csv"
    OUTPUT_DATA_PATH_TEST = f"data/raw/{PROBLEM}/ED_Col_Data_y_test.csv"
elif PROBLEM == "pooling":
    INPUT_DATA_PATH_TRAIN = f"data/raw/{PROBLEM}/Pooling_dataset_x_train.csv"
    OUTPUT_DATA_PATH_TRAIN = f"data/raw/{PROBLEM}/Pooling_dataset_y_train.csv"
    INPUT_DATA_PATH_TEST = f"data/raw/{PROBLEM}/Pooling_dataset_x_test.csv"
    OUTPUT_DATA_PATH_TEST = f"data/raw/{PROBLEM}/Pooling_dataset_y_test.csv"

# Output directories
if PROBLEM in ("function_fitting"):
    TRAINING_DIR = "training_output"  # Base directory for training outputs
elif PROBLEM == "nonconvex_linear" or PROBLEM == "nonconvex_nonlinear":
    TRAINING_DIR = f"training_output/{N_CONSTRAINTS_OPT}"
elif PROBLEM == "extraction_column":
    TRAINING_DIR = f"training_output/{PROBLEM}"
elif PROBLEM == "pooling":
    TRAINING_DIR = f"training_output/{PROBLEM}"
elif PROBLEM == "sin_ineq":
    TRAINING_DIR = f"training_output/{PROBLEM}"
