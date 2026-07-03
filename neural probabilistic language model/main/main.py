from neural_probabilistic_language_model import *
from weight_manager import *

# Initialize the model and weight manager
npblm = NeuralProbabilisticLanguageModel()

npblm(train_model=True, validate_model=True, test_model=True)

# TODO: Weight manager needs to implemented to save and load weights
# weight_manager = WeightManager(filepath="neural probabilistic language model/neural_probabilistic_language_model_weights.pt")

# # Attempt to load existing weights
# weights_exist = weight_manager.load_weights(npblm, device)

# if not weights_exist:
#     print("Training model from scratch...")
#     npblm.train_model()
    
#     # Save the weights so we don't have to train next time
#     weight_manager.save_weights(npblm)


#TODO: Sample from the model