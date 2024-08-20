import numpy as np
import shap
print(shap.__version__)
from kerasAC.interpret.deepshap import combine_mult_and_diffref_1d, create_background

shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
shap.explainers.deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers.deep.deep_tf.passthrough
shap.explainers.deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers.deep.deep_tf.passthrough

def compute_shap_score(model,X_val):

    """
    Inputs:
    model: tf.keras.Model - single array input and single scalar output
    X_val: np.array of shape (1,seq length, feature length) taken by the keras Model

    Output:
    count_explanations: shape - (1,seq length, feature length) shap scores for each input feature
    
    """

    model_wrapper = (model.input, model.output)
    count_explainer=shap.DeepExplainer(model_wrapper,
                                       data=create_background,
                                       combine_mult_and_diffref=combine_mult_and_diffref_1d)
    
    count_explanations=count_explainer.shap_values(X_val)[0]
    return count_explanations 

def predict_classification_proba(X_val,model,weights_files):
    
    preds_all = []
    for wts_file in weights_files:
        model.load_weights(wts_file)
        ypreds = model.predict(X_val,batch_size=128).ravel()
        preds_all.append(ypreds)
    return np.mean(preds_all,axis=0)

def score_classification(X_val,model,weights_files,scoring_fn):
    ## X_val is a single example (shape: 1,input_width,4)
    
    N,input_width,_ = X_val.shape
    score = np.zeros((N,input_width,4))
    for wts_file in weights_files:
        model.load_weights(wts_file)
        
        score += scoring_fn(model,X_val)
    score /= len(weights_files)
    return score
