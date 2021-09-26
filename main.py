from Base.Evaluation.Evaluator import EvaluatorHoldout

from model.scaled_cer import CER, ScaledCER
from scipy import sparse
import pickle


metric_to_optimize = "MAP"
cutoff_list_validation = [5]
cutoff_list_test = [5,15,30] 

ICM_name = 'Genres'   
    
    
train_file_name = 'data/URM/train.npz'
validation_file_name = 'data/URM/validation.npz'
warm_test_file_name = 'data/URM/warm_test.npz'
cold_test_file_name = 'data/URM/cold_test.npz'

URM_train = sparse.load_npz(train_file_name)

URM_validation = sparse.load_npz(validation_file_name)
URM_warm_test = sparse.load_npz(warm_test_file_name)
URM_cold_test = sparse.load_npz(cold_test_file_name)

with open('data/item_indices.pkl', 'rb') as f:
    item_indices = pickle.load(f)
    
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list_validation, ignore_items = item_indices['te.om'])
evaluator_warm_test = EvaluatorHoldout(URM_warm_test, cutoff_list = cutoff_list_test, ignore_items = item_indices['te.om'])
evaluator_cold_test = EvaluatorHoldout(URM_cold_test, cutoff_list = cutoff_list_test, ignore_items = item_indices['tr'])

feature_file_name = 'data/ICM/' + ICM_name + '.npz'
            
ICM = sparse.load_npz(feature_file_name)

scaled_cer = ScaledCER(URM_train, ICM)

hyperparameters = {
    "epochs": 200,
    "k": 50,
    "lu": 0.1,
    "lv": 10.0,
    "le": 1.0e3,
    "a": 1.0,
    "b": 0.01,
    "URM_scaling": 0.217
}

earlystopping_parameters = {
   
    "validation_every_n": 5,
    "stop_on_validation": True,
    "evaluator_object": evaluator_validation,
    "lower_validations_allowed": 5,
    "validation_metric": metric_to_optimize,
}

scaled_cer.fit(**hyperparameters, **earlystopping_parameters)

result_dict, result_string = evaluator_cold_test.evaluateRecommender(scaled_cer)
print("Evaluation results: {}".format(result_string))
