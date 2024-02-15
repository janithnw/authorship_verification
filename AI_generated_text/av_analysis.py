import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import glob
import json
from features import merge_entries, prepare_entry
from utills import chunker, cartesian_product
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import shap
shap.initjs()

class AVAnalysis:
    def __init__(self, model_path, vector_path_prefix=None):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            if len(model) == 4:
                (clf_pan, transformer_pan, scaler_pan, secondary_scaler_pan) = model
            else:
                (
                    aucs,
                    clf_pan,
                    roc_auc,
                    transformer_pan, 
                    scaler_pan,
                    secondary_scaler_pan,
                    feature_sz,
                    train_sz,
                    train_idxs,
                    test_sz,
                    test_idxs
                ) = model
                assert vector_path_prefix is not None
                self.fnames = np.array(transformer_pan.get_feature_names())
                self.XX_train = np.memmap(vector_path_prefix + 'vectorized_XX_train.npy', dtype='float32', mode='r', shape=(train_sz, feature_sz))
                self.XX_test = np.memmap(vector_path_prefix + 'vectorized_XX_test.npy', dtype='float32', mode='r', shape=(test_sz, feature_sz))
                self.feature_sz = feature_sz
                self.train_sz = train_sz
                self.test_sz = test_sz
                self.explainer = shap.LinearExplainer(clf_pan, self.XX_train)
                self.shap_values = self.explainer.shap_values(self.XX_test)
                
            self.clf_pan = clf_pan
            self.transformer_pan = transformer_pan
            self.scaler_pan = scaler_pan
            self.secondary_scaler_pan = secondary_scaler_pan
            
            
    
    def __create_shap_explainer(self, pan_experiment_path):
        with open(f'{pan_experiment_path}experiment_data.p', 'rb') as f:
            (
                aucs,
                clf,
                roc_auc,
                transformer, 
                scaler,
                secondary_scaler,
                feature_sz,
                train_sz,
                train_idxs,
                test_sz,
                test_idxs
            ) = pickle.load(f)

        self.fnames = np.array(transformer.get_feature_names())
        self.XX_train = np.memmap(pan_experiment_path + 'vectorized_XX_train.npy', dtype='float32', mode='r', shape=(train_sz, feature_sz))
        self.XX_test = np.memmap(pan_experiment_path + 'vectorized_XX_test.npy', dtype='float32', mode='r', shape=(test_sz, feature_sz))
        self.explainer = shap.LinearExplainer(clf, XX_train)


    def predict_pan(self, docs_1, docs_2):
        docs_merged_1 = [merge_entries(c) for c in docs_1]
        docs_merged_2 = [merge_entries(c) for c in docs_2]

        X_1 = self.scaler_pan.transform(self.transformer_pan.transform(docs_merged_1).todense())
        X_2 = self.scaler_pan.transform(self.transformer_pan.transform(docs_merged_2).todense())

        p = self.clf_pan.predict_proba(self.secondary_scaler_pan.transform(np.abs(X_1 - X_2)))[:, 1]
        return p
    
    
    def apply_model(self, docs):
        (human_docs_1, human_docs_2, ai_docs_1, ai_docs_2, _) = docs
        
        probs_hh = self.predict_pan(human_docs_1, human_docs_2)
        print(f"Human - Human:          {np.mean(probs_hh):.3f}")
        
        probs_ha1 = self.predict_pan(human_docs_1, ai_docs_1)
        probs_ha2 = self.predict_pan(human_docs_2, ai_docs_2)
        probs_ha = np.concatenate([probs_ha1, probs_ha2])
        print(f"Human - AI:             {np.mean(probs_ha):.3f}")
        
        probs_aa = self.predict_pan(ai_docs_1, ai_docs_2)
        print(f"AI - AI:                {np.mean(probs_aa):.3f}")
        
        shuffled_idxs = np.random.permutation(len(ai_docs_1))
        probs_aa_shuffled = self.predict_pan(
            np.array(ai_docs_1)[shuffled_idxs], 
            np.array(ai_docs_2)
        )
        print(f"AI - AI (Diff Author):  {np.mean(probs_aa_shuffled):.3f}")
        
        return probs_hh, probs_ha, probs_aa, probs_aa_shuffled
    
    def apply_model_multiple_sets(self, all_docs_dict, key_roots):
        # Fond docs that are from the pan dataset records
        common_ids = list(set.intersection(*[set(ids) for _, ids in all_docs_dict.values()]))
        
        # Create a unified dictionary where each key is from {'human_1', 'human_2', 'chatgpt_1', ....}
        # and each value is an ordered list of documents (ordered by the common_ids list)
        res = {}
        for key, (docs, ids) in all_docs_dict.items():
            
            id_to_doc = {i: d for i, d in zip(ids, docs)}
            res[key] = [id_to_doc[i] for i in common_ids]
    
    
        similarity_mat = np.zeros((len(key_roots), len(key_roots)))
        for i in range(len(key_roots)):
            for j in range(i+1):
                if i == j:
                    # Comparing same model, compare between docs_1 and docs_2
                    p = self.predict_pan(res[key_roots[i] + '_1'], res[key_roots[j] + '_2'])
                else:
                    p = self.predict_pan(
                        res[key_roots[i] + '_1'] + res[key_roots[i] + '_2'], 
                        res[key_roots[j] + '_1'] + res[key_roots[j] + '_2']
                    )
                similarity_mat[i][j] = p.mean()
                print(f"{key_roots[i]} - {key_roots[j]}: {p.mean():.3f}")
        return similarity_mat
    
    
    def plot_multiple_set_result(self, res, key_roots):
        res = res.T
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.heatmap(np.where(res == 0, np.nan, res), ax=ax, annot=res, cmap='Blues')
        ax.set_xticklabels(key_roots)
        ax.set_yticklabels(key_roots)
        ax.xaxis.tick_top()
        return ax
    
    
    def plot_shap_summary_av_model(self, n=1000):
        XX_test_sampled = self.XX_test[np.random.choice(self.test_sz, 5000), :]
        shap_values = self.explainer.shap_values(XX_test_sampled)
        shap.summary_plot(shap_values, XX_test_sampled, feature_names=self.fnames, max_display=25)
        plt.tight_layout()
        
    def get_shap_values(self, docs_1, docs_2):
        X_1 = self.scaler_pan.transform(
            self.transformer_pan.transform(
                [merge_entries(d) for d in docs_1]
            ).todense()
        )
        X_2 = self.scaler_pan.transform(
            self.transformer_pan.transform(
                [merge_entries(d) for d in docs_2]
            ).todense()
        )

        X_diff = self.secondary_scaler_pan.transform(np.abs(X_1 - X_2))
        shap_values = self.explainer.shap_values(X_diff)
        return shap_values
    
    
    def plot_shap_summary(self, docs_1, docs_2):
        plt.clf()
        X_1 = self.scaler_pan.transform(
            self.transformer_pan.transform(
                [merge_entries(d) for d in docs_1]
            ).todense()
        )
        X_2 = self.scaler_pan.transform(
            self.transformer_pan.transform(
                [merge_entries(d) for d in docs_2]
            ).todense()
        )

        X_diff = self.secondary_scaler_pan.transform(np.abs(X_1 - X_2))
        shap_values = self.explainer.shap_values(X_diff)
        shap.summary_plot(shap_values, X_diff, feature_names=self.fnames, max_display=25)
        plt.tight_layout()
        plt.show()
        
    
    def get_all_shap_values(self, docs):
        (human_docs_1, human_docs_2, ai_docs_1, ai_docs_2, pair_ids) = docs
        shap_hh = self.get_shap_values(human_docs_1, human_docs_2)
        shap_ha_1 = self.get_shap_values(human_docs_1, ai_docs_1)
        shap_ha_2 = self.get_shap_values(human_docs_2, ai_docs_2)
        shap_ha = np.concatenate([shap_ha_1, shap_ha_2])
        shap_aa = self.get_shap_values(ai_docs_1, ai_docs_2)
        return (shap_hh, shap_ha, shap_aa)
    
    
    def get_all_shap_summary(self, docs, model_name):
        (shap_hh, shap_ha, shap_aa) = self.get_all_shap_values(docs)
        shap_summaries_df = pd.DataFrame(
            data=np.vstack([
                np.abs(shap_hh.mean(axis=0)), 
                np.abs(shap_ha.mean(axis=0)),
                np.abs(shap_aa.mean(axis=0)),
            ]).T,
            columns=['human-human', f'{model_name}-{model_name}', f'human-{model_name}']
        )
        shap_summaries_df['fnames'] = self.fnames
        shap_summaries_df = shap_summaries_df.set_index('fnames')
        return shap_summaries_df
    
    def plot_shap_summary_single_comparison(self, shap_values: pd.Series, limit=15):
        fig, ax = plt.subplots( figsize=(5, 12))
        shap_values = shap_values.sort_values(ascending=False).head(limit).sort_values()
        x = np.arange(len(shap_values))
        x_labels = shap_values.index
        ax.barh(x, shap_values)
        ax.set_yticks(x)
        ax.set_yticklabels(labels=x_labels, rotation=0)
        ax.set_xlabel('SHAP Value')
        plt.tight_layout()
        
        return
    
    
    def plot_shap_summary_two_comparisons(self, shap_summaries_df, col1, col2, limit=15):
        buffer = 3
        shap_summaries_df['importance_diff'] = shap_summaries_df[col2] - shap_summaries_df[col1]

        shap_summaries_selected_df = pd.concat([
            shap_summaries_df.sort_values('importance_diff', ascending=False).head(limit),
            # NaNs as buffer
            pd.DataFrame({c: [np.nan]*buffer for c in shap_summaries_df.columns}, index=[f'buffer_{i}' for i in range(buffer)]),
            shap_summaries_df.sort_values('importance_diff', ascending=False).tail(limit)
        ])

        x = np.arange(len(shap_summaries_selected_df))
        x_labels = [
            f if 'buffer' not in f else ' ' for f in shap_summaries_selected_df.index
        ]

        fig, ax = plt.subplots( figsize=(5,12))
        width = 0.4
        offset = -width/2
        ax.barh(x + offset, shap_summaries_selected_df[col1], height=width, label=col1)
        offset = width/2
        ax.barh(x + offset, shap_summaries_selected_df[col2], height=width, label=col2)
        plt.axhline(y=limit - width, color='gray')
        plt.axhline(y=limit + buffer - width - offset, color='gray')
        ax.set_yticks(x)
        ax.set_yticklabels(labels=x_labels, rotation=0)
        ax.set_xlabel('SHAP Value')
        ax.legend()
        plt.tight_layout()
    
    
    
def unfix_quotes(doc):
    """
    Turn single quotes to double
    """
    text =  ' '.join([e['preprocessed'] for e in doc])
    text = text.replace('\'', '"')
    return [prepare_entry(text, mode='accurate', tokenizer='casual')]

def load_generated_docs(path):
    human_docs_1 = []
    human_docs_2 = []
    ai_docs_1 = []
    ai_docs_2 = []

    pair_ids = []
    for fname in glob.glob(path):
        with open(fname, 'r') as f:
            for l in f:
                d = json.loads(l)
                human_docs_1.append(unfix_quotes(d['pair'][0]['human']))
                human_docs_2.append(unfix_quotes(d['pair'][1]['human']))

                ai_docs_1.append(unfix_quotes(d['pair'][0]['ai'][:20]))
                ai_docs_2.append(unfix_quotes(d['pair'][1]['ai'][:20]))
                pair_ids.append(d['id'])
                
    print('Read:', len(human_docs_1))
    print("Avg Lengths:")
    print(f"Human 1:    {np.mean([len(d[0]['tokens']) for d in human_docs_1]):.2f}")
    print(f"Human 2:    {np.mean([len(d[0]['tokens']) for d in human_docs_2]):.2f}")
    print(f"AI 1:       {np.mean([len(d[0]['tokens']) for d in ai_docs_1]):.2f}")
    print(f"AI 2:       {np.mean([len(d[0]['tokens']) for d in ai_docs_2]):.2f}")
    return human_docs_1, human_docs_2, ai_docs_1, ai_docs_2, pair_ids



