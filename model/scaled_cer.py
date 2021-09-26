import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import tensorflow as tf
import pickle
import scipy as sp


from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


class CER(BaseRecommender, Incremental_Training_Early_Stopping):
    
    RECOMMENDER_NAME = "CER"
    AVAILABLE_CONFIDENCE_SCALING = [None, "linear", "log"]
    
    def __init__(self, URM_train, ICM_train):
        
        super(CER, self).__init__(URM_train)
        
        self.ICM_train = ICM_train.copy()
        
        self.ICM_train.eliminate_zeros()
        
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)
            
        assert self.USER_factors.shape[0] > user_id_array.max(),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], user_id_array.max())
            
        
        Fe = np.dot(self.ICM_train, self.E)
        ITEM_factors = self.ITEM_factors.copy()
        
        ITEM_factors[self.cold_items,:] = Fe[self.cold_items,:]
        del Fe
                
                
        if(items_to_compute is not None):
            item_scores = -np.ones((len(user_id_array), ITEM_factors.shape[0]), dtype = np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], ITEM_factors[items_to_compute, :].T)
        
        else:
            item_scores = np.dot(self.USER_factors[user_id_array], ITEM_factors.T)
            
        
        return item_scores
    
    def fit(self, epochs = 200, k = 50, lu = 0.1, lv = 10.0, le = 1.0e3, a = 1.0, b = 0.01, confidence_scaling = None, alpha = 1.0, epsilon = 1.0, 
            init_factors_CER = True, URM_scaling = 1.0, **earlystopping_kwargs):
        
        self.k = k
        self.lu = lu
        self.lv = lv
        self.le = le
        
       
       
        self.URM_scaling = URM_scaling
       
       
        self.URM = self.get_URM_train()
        
            
        self.ICM_train = self.ICM_train.toarray()
        self.b = 1.0
        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
           raise ValueError("Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))
        
        print("Using confidence scaling {}".format(confidence_scaling))
        
        if(confidence_scaling is None):
            
            self.a = a
            self.b = b
            
        elif(confidence_scaling == 'linear'):
            
            self.URM.data = 1.0 + alpha*self.URM.data
            
            
        else:
            
            self.URM.data = 1.0 + alpha * np.log(1.0 + self.URM.data/epsilon)
            
        
        self.URM_train_csc = self.URM.copy().tocsc()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        
        self.n_users, self.n_items = self.URM.shape
        self.E = np.random.randn(self.ICM_train.shape[1], self.k).astype(np.float32)
        
        if(init_factors_CER):
            self.USER_factors = np.random.rand(self.n_users, self.k).astype(np.float32)
            self.ITEM_factors = np.random.rand(self.n_items, self.k).astype(np.float32)
        else:
            self.USER_factors = self._init_factors(self.n_users, False) 
            self.ITEM_factors = self._init_factors(self.n_items)
        
        warm_user_mask = np.ediff1d(self.URM.indptr) > 0
        warm_item_mask = np.ediff1d(self.URM.tocsc().indptr) > 0
        cold_item_mask = np.ediff1d(self.URM.tocsc().indptr) == 0
        
        self.u_rated = np.arange(0, self.n_users, dtype = np.int32)[warm_user_mask]
        self.i_rated = np.arange(0, self.n_items, dtype = np.int32)[warm_item_mask]
        self.cold_items = np.arange(0, self.n_items, dtype = np.int32)[cold_item_mask]
        
        self._update_best_model()
        
        self.loss = np.exp(50)
        self.Ik = np.eye(self.k, dtype = np.float32)
        self.FF = self.lv * np.dot(self.ICM_train.T, self.ICM_train) + self.le * np.eye(self.ICM_train.shape[1])
        self._train_with_early_stopping(epochs, algorithm_name = self.RECOMMENDER_NAME, **earlystopping_kwargs)
        del self.URM_train_csc, self.Ik, self.FF
        self.USER_factors = self.USER_factors_best.copy()
       
        self.ITEM_factors = self.ITEM_factors_best.copy()
       
        self.E = self.E_best.copy()
    
    def _check_convergence_based_on_loss(self):
        
        cond = np.abs(self.loss_old - self.loss)/ self.loss_old
        return cond < 1e-4
    
    def _prepare_model_for_validation(self):
        pass
    
    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.E_best = self.E.copy()
        
        
    def _run_epoch(self, num_epoch):
        
        Fe = np.dot(self.ICM_train, self.E)
        self.loss_old = self.loss
        self.loss = 0
        
        Vr = self.ITEM_factors[self.i_rated,:]
        XX = np.dot(Vr.T, Vr) * self.b + self.Ik * self.lu
        del Vr
        for i in range(self.n_users):
            
            if(i in self.u_rated):
                start_pos = self.URM.indptr[i]
                end_pos = self.URM.indptr[i + 1]
                
                user_profile = self.URM.indices[start_pos:end_pos]
                user_confidence = self.URM.data[start_pos: end_pos]
                
                user_confidence1 = user_confidence - self.b
                
                Vi = self.ITEM_factors[user_profile, :]
                self.USER_factors[i, :] = np.linalg.solve(np.dot(Vi.T * user_confidence1, Vi) + XX, np.sum(Vi * user_confidence.reshape((1,len(user_confidence))).T, axis = 0))
               
            self.loss += 0.5 * self.lu * np.sum(self.USER_factors[i, :] ** 2)
         
        Ur = self.USER_factors[self.u_rated, :]
        XX = np.dot(Ur.T, Ur) * self.b
        del Ur
        for j in range(self.n_items):
            B = XX.copy()
            
            if(j in self.i_rated):
                start_pos = self.URM_train_csc.indptr[j]
                end_pos = self.URM_train_csc.indptr[j + 1]
                item_profile = self.URM_train_csc.indices[start_pos:end_pos]
                item_confidence = self.URM_train_csc.data[start_pos:end_pos]
                
                item_confidence1 = item_confidence - self.b
                
                
                Uj = self.USER_factors[item_profile, :]
                B += np.dot(Uj.T * item_confidence1, Uj)
            
                self.ITEM_factors[j, :] = np.linalg.solve(B + self.Ik * self.lv, np.sum(Uj * item_confidence.reshape((1, len(item_confidence))).T, axis = 0) + Fe[j, :] * self.lv)
                self.loss += 0.5 * np.linalg.multi_dot((self.ITEM_factors[j,:], B, self.ITEM_factors[j, :]))
                self.loss += 0.5 * sum(np.ones((len(item_profile))) * item_confidence)
            
                self.loss -= np.sum(np.multiply(Uj * item_confidence.reshape((1, len(item_confidence))).T, self.ITEM_factors[j, :]))
                
            else:
                self.ITEM_factors[j, :] = np.linalg.solve(B + self.Ik * self.lv, Fe[j, :] * self.lv)
                
                
            self.loss += 0.5 * self.lv * np.sum((self.ITEM_factors[j, :] - Fe[j, :]) ** 2)
            
        self.E = np.linalg.solve(self.FF, self.lv * np.dot(self.ICM_train.T, self.ITEM_factors))
        self.loss += 0.5 * self.le * np.sum(self.E ** 2)
        
    def _init_factors(self, num_factors, assign_values=True):

        if assign_values:
            return self.k**-0.5*np.random.random_sample((num_factors, self.k))

        else:
            return np.empty((num_factors, self.k))

        
    def save_model(self, folder_path, file_name = None):
       
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
            
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        
        data_dict_to_save = {"USER_factors": self.USER_factors,
                             "ITEM_factors": self.ITEM_factors,
                             "E": self.E}
        
        pickle.dump(data_dict_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        
        self._print("Saving complete")
        
        
                    
                    
class ScaledCER(CER):
    
    RECOMMENDER_NAME = f'scaled{CER.RECOMMENDER_NAME}'
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        

    def rescale_URM_matrix(self, URM_train):
        
        if self.URM_scaling == 1.0:
            
            scaled_URM_train = URM_train
            
        else:
        
            norm = norm = np.sqrt(URM_train.getnnz(axis=0)) 
            
            scaling_values = np.power(norm, self.URM_scaling-1, where=norm != 0)
            scaling_matrix = sp.sparse.diags(scaling_values)
            
            scaled_URM_train = URM_train.dot(scaling_matrix)
        
        return scaled_URM_train
            
    def get_URM_train(self):
    
        scaled_URM_train = super().get_URM_train()
    
        scaled_URM_train = self.rescale_URM_matrix(scaled_URM_train)
        
        return scaled_URM_train