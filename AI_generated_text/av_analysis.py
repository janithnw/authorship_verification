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

class AVAnalysis:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            (clf_pan, transformer_pan, scaler_pan, secondary_scaler_pan) = pickle.load(f)
            self.clf_pan = clf_pan
            self.transformer_pan = transformer_pan
            self.scaler_pan = scaler_pan
            self.secondary_scaler_pan = secondary_scaler_pan
    
    def predict_pan(self, docs_1, docs_2):
        docs_merged_1 = [merge_entries(c) for c in docs_1]
        docs_merged_2 = [merge_entries(c) for c in docs_2]

        X_1 = self.scaler_pan.transform(self.transformer_pan.transform(docs_merged_1).todense())
        X_2 = self.scaler_pan.transform(self.transformer_pan.transform(docs_merged_2).todense())

        p = self.clf_pan.predict_proba(self.secondary_scaler_pan.transform(np.abs(X_1 - X_2)))[:, 1]
        return p
    
    
    def apply_model(self, docs):
        (human_docs_1, human_docs_2, ai_docs_1, ai_docs_2) = docs
        
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



