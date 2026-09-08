import numpy as np
import re

_FOLDINGS = {
    'AO': 'AA',
    'AX': 'AH',
    'AX-H': 'AH',
    'AXR': 'ER',
    'HV': 'HH',
    'IX': 'IH',
    'EL': 'L',
    'EM': 'M',
    'EN': 'N',
    'NX': 'N',
    'ENG': 'NG',
    'PCL': 'P',
    'TCL': 'T',
    'KCL': 'K',
    'BCL': 'B',
    'DCL': 'D',
    'GCL': 'G',
    'H#': 'SIL',
    'PAU': 'SIL',
    'EPI': 'SIL',
    'UX': 'UW'       
}

def align(seq, probs):

    def distance(r, c):
        category = seq[r]
        timestep = c
        return probs[timestep, category]
        
    nrow = len(seq)
    ncol = probs.shape[0] # time steps
    
    M = warp(range(nrow), range(ncol), distance)
    
    alignment = traceback(M[1:, 1:], seq)
    return alignment, M[1:, 1:]
    
def warp(S, T, d):
    
    nrow = len(S)
    ncol = len(T)
    
    M = np.full((nrow+1, ncol+1), np.inf)
    
    M[0, 0] = 0
    
    # Some optimization is being left on the table; this should be computable
    # for a whole column at a time
    for r in range(1, nrow+1):
        for c in range(1, ncol+1):
            # need -1 offset for cost because of extra first row
            # and first column in M that isn't present in the
            # `probs` matrix
            cost = d(r-1, c-1)
            M[r, c] = cost + min(M[r, c-1], M[r-1, c-1])
            
    return M
    
def traceback(M, sequence):

    nrow, ncol = M.shape
    seq = [nrow-1]
    
    r = nrow-1
    c = ncol-1
    
    curr_prob = M[r, c]
    
    while c >= 0:
        if r == 0:
            seq += [0 for x in range(ncol - len(seq))]
            break
        
        if M[r-1, c-1] > M[r, c-1]:
            seq.append(r)
            c -= 1
        else:
            c -= 1
            r -= 1
            
            seq.append(r)
            
    rs = seq[::-1]
    rs = [sequence[r] for r in rs]
    
    return rs
    
def collapse(s):
    a = [s[0]]
    for symbol in s[1:]:
        if a[-1] != symbol:
            a.append(symbol)
    return a
    
def fold_phone(p):

    stress_mark = re.findall(r'\d+', p)
    if stress_mark:
        stress_mark = stress_mark[0]
    else:
        stress_mark = ''
    phone_label = re.sub(r'\d', '', p)
    if phone_label in _FOLDINGS:
        phone_label = _FOLDINGS[phone_label]
    return phone_label + stress_mark

    
def load_dictionary(dname, cmuformat=True):

    

    mapping = dict()
    
    if cmuformat:

        with open(dname, 'r') as d:
            for line in d:
                line = line.strip()
                if not line or line.startswith(';;;'):
                    continue
                all_items = line.split()
                if len(all_items) < 2:
                    continue
                word = all_items[0]
                word = re.sub(r'\(\d*\)', '', word)
                # NLTK cmudict uses ``WORD <variant_index> PHONE ...``; skip the index.
                pron_start = 2 if len(all_items) >= 3 and all_items[1].isdigit() else 1
                pronunciation = all_items[pron_start:]
                pronunciation = [fold_phone(p) for p in pronunciation]
                if not pronunciation:
                    continue
                
                if word not in mapping:
                    mapping[word] = [pronunciation]
                else:
                    mapping[word].append(pronunciation)
                    
        mapping['sil'] = [['H#'], []]
        return mapping
        
    else:
        print('Custom dictionary formats not supported yet. Please format use CMU dict formatting and run again.')
        sys.exit(1)
                
