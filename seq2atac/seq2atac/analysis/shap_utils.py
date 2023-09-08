import numpy as np
from seq2atac.analysis.enrichment_utils import get_alt_sequence, get_refalt_sequence
from seq2atac.stable import one_hot_encode
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import shap
print(shap.__version__)
from kerasAC.interpret.deepshap import combine_mult_and_diffref_1d, create_background
from kerasAC.interpret.profile_shap import create_background_atac, combine_mult_and_diffref_atac
from kerasAC.vis import plot_seq_importance
from kerasAC.interpret.profile_shap import create_explainer
from seq2atac.viz import plot_weights_given_ax

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

# def compute_shap_score_seq(model,X_val):
#     model_wrapper = (model.input, model.output[0])
#     count_explainer=shap.DeepExplainer(model_wrapper,
#                                        data=create_background,
#                                        combine_mult_and_diffref=combine_mult_and_diffref_1d)
    
#     count_explanations=count_explainer.shap_values(X_val)[0]
#     return count_explanations 


# def compute_shap_score_profile(model,X_val):
#     prof_output = model.output[1] # B X 324
#     logits = prof_output - tf.reduce_mean(prof_output, axis=1, keepdims=True)

#     # Stop gradients flowing to softmax, to avoid explaining those
#     logits_stopgrad = tf.stop_gradient(logits)
#     probs = tf.nn.softmax(logits_stopgrad, axis=1)

#     logits_weighted = logits * probs  # Shape: B x 324
#     prof_sum = tf.reduce_sum(logits_weighted, axis=1)

#     count_explainer = shap.DeepExplainer(
#         model=(model.input, prof_sum),
#         data=create_background_atac,
#         combine_mult_and_diffref=combine_mult_and_diffref_atac
#     )
    
#     count_explanations=count_explainer.shap_values(X_val)[0]
#     return count_explanations 

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

def compute_grad_x_input(model,input_batch):
    ### Compute grad x input
    # model: tf.keras.model
    # input_batch: np.array: (num_samples, window_size, 4), or anything that the model can process
    X_tensor = tf.convert_to_tensor(input_batch)
    with tf.GradientTape() as t:
        t.watch(X_tensor)
        y_pred = model(X_tensor)

    dy_dx = t.gradient(y_pred, X_tensor)

    importance = tf.math.multiply(dy_dx, X_tensor)

    imp_np = importance.numpy()
    return imp_np

def plot_summit_centered(interest,fasta_seq,cancer_name,model,weights_files,flank=250,outdir=None,mutation_in_middle=False):

    for idx,row in interest.iterrows():
        chrom,pos,ref,alt=row["Chromosome"],row["hg38_start"],row["Reference_Allele"],row["Tumor_Seq_Allele2"]
        X_ref,X_alt = [one_hot_encode(x) for x in get_alt_sequence(interest.loc[[idx],:],1364,fasta_seq)]
        
        # assert flank <= 250
        raw_dist = row["hg38_start"] - (row["peak_start"] + row["peak_end"])//2
        print(raw_dist)
        dist_from_summit = flank + raw_dist
        print(dist_from_summit)

        # set up plot
        fig,axes = plt.subplots(2,1,figsize=(20*flank/50,8))
        mid_pt = 1364//2 
        if mutation_in_middle:
            mid_pt += raw_dist
            dist_from_summit = flank
        plt_start = mid_pt-flank
        plt_end = mid_pt+flank

        # compute score
        score1 = score_classification(X_ref,model,weights_files,compute_shap_score) * X_ref
        # plot importance
        plot_weights_given_ax(axes[0],score1[0][plt_start:plt_end], subticks_frequency=20)
        prob_recalc = predict_classification_proba(X_ref, model, weights_files)[0]
        if "proba_ref_summit_centered" in row:
            df_value = row["proba_ref_summit_centered"]
            assert np.isclose(prob_recalc,df_value,atol=1e-2), f"{prob_recalc},{df_value}"
        axes[0].set_title(f"REF, proba: {prob_recalc}")
        axes[0].axvspan(dist_from_summit, dist_from_summit+1, alpha=0.2, color='gray')

    
        # compute score
        score2 = score_classification(X_alt,model,weights_files,compute_shap_score) * X_alt
        # plot importance
        plot_weights_given_ax(axes[1],score2[0][plt_start:plt_end], subticks_frequency=20)
        prob_recalc = predict_classification_proba(X_alt, model, weights_files)[0]
        if "proba_alt_summit_centered" in row:
            df_value = row["proba_alt_summit_centered"]
            assert np.isclose(prob_recalc,row["proba_alt_summit_centered"],atol=1e-2), f"{prob_recalc},{df_value}"
        axes[1].set_title(f"ALT, proba: {prob_recalc}")
        axes[1].axvspan(dist_from_summit, dist_from_summit+1, alpha=0.2, color='gray')
        
        
        ymax = max(score1[0][plt_start:plt_end].max(),score2[0][plt_start:plt_end].max())
        axes[0].set_ylim(-ymax,ymax)
        axes[1].set_ylim(-ymax,ymax)
        
        fig_title = f"{cancer_name} {chrom}:{pos}-{ref}/{alt}"
        if "gene" in row:
            gene = row["gene"]
            fig_title = f"{cancer_name}-{idx} {chrom}:{pos}-{ref}/{alt}: {gene}"
        fig.suptitle(fig_title)
        fig.tight_layout()
        if outdir:
            plt.savefig(f"{outdir}/{idx}.png")
        else:
            plt.show()


def plot_mutation_centered(interest,fasta_seq,cancer_name,model,weights_files,flank=250,outdir=None):

    for idx,row in interest.iterrows():
        chrom,pos,ref,alt=row["Chromosome"],row["hg38_start"],row["Reference_Allele"],row["Tumor_Seq_Allele2"]
        X_ref,X_alt = [one_hot_encode(x) for x in get_refalt_sequence(interest.loc[[idx],:],1364,fasta_seq)]
        
        # assert flank <= 250

        # set up plot
        fig,axes = plt.subplots(2,1,figsize=(20*flank/50,8))
        mid_pt = 1364//2
        plt_start = mid_pt-flank
        plt_end = mid_pt+flank

        # compute score
        score1 = score_classification(X_ref,model,weights_files,compute_shap_score) * X_ref
        # plot importance
        plot_weights_given_ax(axes[0],score1[0][plt_start:plt_end], subticks_frequency=20)
        prob_recalc = predict_classification_proba(X_ref, model, weights_files)[0]
        if "proba_ref_mutation_centered" in row:
            df_value = row["proba_ref_mutation_centered"]
            assert np.isclose(prob_recalc,row["proba_ref_mutation_centered"],atol=1e-2), f"{prob_recalc},{df_value}"
        axes[0].set_title(f"REF, proba: {prob_recalc}")
        axes[0].axvspan(flank, flank+1, alpha=0.2, color='gray')

    
        # compute score
        score2 = score_classification(X_alt,model,weights_files,compute_shap_score) * X_alt
        # plot importance
        plot_weights_given_ax(axes[1],score2[0][plt_start:plt_end], subticks_frequency=20)
        prob_recalc = predict_classification_proba(X_alt, model, weights_files)[0]
        if "proba_alt_mutation_centered" in row:
            df_value = row["proba_alt_mutation_centered"]
            assert np.isclose(prob_recalc,row["proba_alt_mutation_centered"],atol=1e-2), f"{prob_recalc},{df_value}"
        axes[1].set_title(f"ALT, proba: {prob_recalc}")
        axes[1].axvspan(flank, flank+1, alpha=0.2, color='gray')
        
        
        ymax = max(score1[0][plt_start:plt_end].max(),score2[0][plt_start:plt_end].max())
        axes[0].set_ylim(-ymax,ymax)
        axes[1].set_ylim(-ymax,ymax)
        
        fig_title = f"{cancer_name} {chrom}:{pos}-{ref}/{alt}"
        if "gene" in row:
            gene = row["gene"]
            fig_title = f"{cancer_name}-{idx} {chrom}:{pos}-{ref}/{alt}: {gene}"
        fig.suptitle(fig_title)
        fig.tight_layout()
        if outdir:
            plt.savefig(f"{outdir}/{idx}.png")
        else:
            plt.show()

def plot_peak(interest,fasta_seq,cancer_name,model,weights_files,flank=250,outdir=None):
    
    input_size = 1364

    for idx,row in interest.iterrows():
        chrom,start,end = row["seqnames"], row["start"], row["end"]
        
        summit = (start+end)//2
        start = summit - input_size//2
        end = start + input_size
        X_ref = one_hot_encode([fasta_seq[chrom][start:end]])
        
        # set up plot
        fig,axes = plt.subplots(figsize=(30*flank/100,3))
        mid_pt = 1364//2
        plt_start = mid_pt-flank
        plt_end = mid_pt+flank

        # compute score
        score1 = score_classification(X_ref,model,weights_files,compute_shap_score) * X_ref
        # plot importance
        plot_weights_given_ax(axes,score1[0][plt_start:plt_end], subticks_frequency=20)
        prob_recalc = predict_classification_proba(X_ref, model, weights_files)[0]
        axes.set_title(f"proba: {prob_recalc}")
        
        if "motif_start" in row and "motif_end" in row:
            motif_start, motif_end = row["motif_start"], row["motif_end"]
            motif_len = motif_end - motif_start
            motif_relative_to_summit = motif_start - (summit - flank)
            axes.axvspan(motif_relative_to_summit, motif_relative_to_summit+motif_len, alpha=0.2, color='gray')
        
        ymax = score1[0][plt_start:plt_end].max()
        axes.set_ylim(-ymax,ymax)
        
        fig_title = f"{cancer_name} {chrom}:{start}-{end}"
        if "gene" in row:
            gene = row["gene"]
            fig_title = f"{cancer_name}-{idx} {chrom}:{start}-{end} - {gene}"
        fig.suptitle(fig_title)
        fig.tight_layout()
        if outdir:
            plt.savefig(f"{outdir}/{idx}.png")
        else:
            plt.show()
