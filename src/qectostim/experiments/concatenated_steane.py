import stim
import random
import numpy as np
import json
import sys

def append_noisy_wait(circuit,list_loc,N,p,steps=1):
    ew =  3/4*(1-(1-4/3*gamma)**steps)
    for i in list_loc:
        for j in range(N):
            circuit.append("DEPOLARIZE1", i+j, ew)

def reset(circuit,loc,N):
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
        
def noisy_reset(circuit,loc,N,p):
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
    for i in range(N):
        circuit.append("X_ERROR", loc+i, p)

def append_h(circuit, loc, N_prev, N_now):
    if N_prev==1:
        for i in range(N_now):
            circuit.append("H", loc+i)
    else:
        for i in range(N_now):
            append_h(circuit, (loc+i)*N_prev, 1, N_prev)
def append_transversal_h(circuit, loc, N_prev, N_now):
    if N_prev==1:
        for i in range(N_now):
            circuit.append("H", loc+i)
    else:
        for i in range(N_now):
            append_h(circuit, (loc+i)*N_prev, 1, N_prev)
            
def append_swap(circuit, loc1, loc2, N_prev, N_now):
    for i in range(N_prev*N_now):
        circuit.append("SWAP", [N_prev*loc1+i, N_prev*loc2+i])

    
def append_cnot(circuit, loc1, loc2, N_prev, N_now):
    N=N_prev*N_now
    for i in range(N):
        circuit.append("CNOT", [loc1*N_prev+i, loc2*N_prev+i])
        
def append_noisy_cnot(circuit, loc1, loc2, N_prev, N_now, p):
    N=N_prev*N_now
    for i in range(N):
        circuit.append("CNOT", [loc1*N_prev+i, loc2*N_prev+i])
    for i in range(N):
        circuit.append("DEPOLARIZE2", [loc1*N_prev+i,loc2*N_prev+i], p)

def append_m(circuit, loc, N_prev, N_now):
    global detector_now
    
    if N_prev==1:
        for i in range(N_now):
            circuit.append("M", loc+i)
        for i in range(N_now):
            circuit.append("DETECTOR",stim.target_rec(i-N_now))
        detector_m=[detector_now, detector_now+N_now]
        detector_now+=N_now
    else:
        detector_m=[append_m(circuit, (loc+i)*N_prev, 1, N_prev) for i in range(N_now)]
    
    return detector_m

def append_noisy_m(circuit, loc,N_prev,N_now, p):
    global detector_now
    
    if N_prev==1:
        for i in range(N_now):
            circuit.append("X_ERROR", loc+i, p)
            circuit.append("M", loc+i)
        for i in range(N_now):
            circuit.append("DETECTOR",stim.target_rec(i-N_now))
        detector_m=[detector_now, detector_now+N_now]
        detector_now+=N_now
    else:
        detector_m=[append_noisy_m(circuit, (loc+i)*N_prev, 1, N_prev, p) for i in range(N_now)]
    
    return detector_m
        
def append_0prep(circuit, loc1, N_prev, N_now):
    if N_prev==1:
        reset(circuit,loc1,N_now)
    else:
        for i in range(N_now):
            append_0prep(circuit, (loc1+i)*N_prev, 1, N_prev)
    if N_now==N_steane:
        append_h(circuit,(loc1+0)*N_prev,1,N_prev)
        append_h(circuit,(loc1+1)*N_prev,1,N_prev)
        append_h(circuit,(loc1+3)*N_prev,1,N_prev)
        
        append_cnot(circuit, (loc1+1)*N_prev, (loc1+2)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+3)*N_prev, (loc1+5)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+0)*N_prev, (loc1+4)*N_prev,1,N_prev)
        
        append_cnot(circuit, (loc1+1)*N_prev, (loc1+6)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+3)*N_prev, (loc1+4)*N_prev,1,N_prev)
        
        append_cnot(circuit, (loc1+1)*N_prev, (loc1+5)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+4)*N_prev, (loc1+6)*N_prev,1,N_prev)

def append_noisy_0prep(circuit, loc1, loc2, N_prev, N_now, p):
    global detector_now
    n_now=N_now
    
    if N_prev==1:
        noisy_reset(circuit,loc1,N_now,p)
        noisy_reset(circuit,loc2,N_now,p)
        detector_0prep = []
    else:
        detector_0prep = []
        for i in range(n_now):
            detector_0prep.append(append_noisy_0prep(circuit, (loc1+i)*N_prev, (loc1+n_now+i)*N_prev, 1, N_prev, p))
        if N_now==N_steane:
            detector_0prep.append(append_noisy_0prep(circuit, (loc2)*N_prev, (loc2+n_now)*N_prev, 1, N_prev, p))
        else:
            for i in range(n_now):
                detector_0prep.append(append_noisy_0prep(circuit, (loc2+i)*N_prev, (loc2+n_now+i)*N_prev, 1, N_prev, p))
    if N_now==N_steane and N_prev!=1:
        detector_X=[]
        detector_Z=[]
        
        append_h(circuit,(loc1+0)*N_prev,1,N_prev)
        append_h(circuit,(loc1+1)*N_prev,1,N_prev)
        append_h(circuit,(loc1+3)*N_prev,1,N_prev)
        
        for a in range(6):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+2)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+5)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+4)*N_prev,1,N_prev, p)
        
        for a in range(7):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+6)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+4)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+5)*N_prev],N_prev,p,steps=1)
        
        for a in range(7):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc2)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
        detector_0prep.extend(detector_0prep_c4)
        detector_X.append(detector_X_c4)
        detector_Z.append(detector_Z_c4)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+5)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc1+6)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+2)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev, (loc1+3)*N_prev],N_prev,p,steps=1)
        
        for a in range(7):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc2)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
        detector_0prep.extend(detector_0prep_c4)
        detector_X.append(detector_X_c4)
        detector_Z.append(detector_Z_c4)
            
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev,(loc1+1)*N_prev,(loc1+2)*N_prev,(loc1+3)*N_prev,(loc1+5)*N_prev,(loc1+6)*N_prev],N_prev,p,steps=1)
        
        for a in range(7):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc2)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
        detector_0prep.extend(detector_0prep_c4)
        detector_X.append(detector_X_c4)
        detector_Z.append(detector_Z_c4)
        
        append_noisy_cnot(circuit, (loc1+5)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev,(loc1+1)*N_prev,(loc1+2)*N_prev,(loc1+3)*N_prev,(loc1+4)*N_prev,(loc1+6)*N_prev],N_prev,p,steps=1)
        
        for a in range(7):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc1+a)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
            detector_0prep.extend(detector_0prep_c4)
            detector_X.append(detector_X_c4)
            detector_Z.append(detector_Z_c4)
        detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit, (loc2)*N_prev,(loc2+1)*N_prev,(loc2+2)*N_prev, (loc2+3)*N_prev, 1, N_prev, p)
        detector_0prep.extend(detector_0prep_c4)
        detector_X.append(detector_X_c4)
        detector_Z.append(detector_Z_c4)
        
        detector_0prep_l2 = append_noisy_m(circuit,(loc2)*N_prev,1,N_prev,p)
        
        return detector_0prep, detector_0prep_l2, detector_X, detector_Z
    
    elif N_now==N_steane:
        append_h(circuit,(loc1+0)*N_prev,1,N_prev)
        append_h(circuit,(loc1+1)*N_prev,1,N_prev)
        append_h(circuit,(loc1+3)*N_prev,1,N_prev)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+2)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+5)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+4)*N_prev,1,N_prev, p)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+6)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+4)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+5)*N_prev],N_prev,p,steps=1)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+5)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc1+6)*N_prev,1,N_prev, p)
        append_noisy_cnot(circuit, (loc1+2)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev, (loc1+3)*N_prev],N_prev,p,steps=1)
        
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev,(loc1+1)*N_prev,(loc1+2)*N_prev,(loc1+3)*N_prev,(loc1+5)*N_prev,(loc1+6)*N_prev],N_prev,p,steps=1)
        
        append_noisy_cnot(circuit, (loc1+5)*N_prev, (loc2)*N_prev,1,N_prev, p)
        append_noisy_wait(circuit,[(loc1+0)*N_prev,(loc1+1)*N_prev,(loc1+2)*N_prev,(loc1+3)*N_prev,(loc1+4)*N_prev,(loc1+6)*N_prev],N_prev,p,steps=1)
        
        detector_0prep.append(append_noisy_m(circuit,(loc2)*N_prev,1,N_prev,p))
        append_noisy_wait(circuit,[(loc1+0)*N_prev,(loc1+1)*N_prev,(loc1+2)*N_prev,(loc1+3)*N_prev,(loc1+4)*N_prev,(loc1+6)*N_prev],N_prev,p,steps=1)
        
        return detector_0prep


def append_noisy_0prep_v2(circuit, loc1, loc2, N_prev, N_now, p):
    """
    Corrected L2 Steane prep following the C4/C6 pattern:
    1. NO EC rounds during prep (prevents noise amplification)
    2. Decorrelation CNOTs after verification measurement
    
    This is fundamentally different from the original which had 45 EC rounds
    and no decorrelation, causing ~50% error rate.
    """
    global detector_now
    n_now = N_now
    
    if N_prev == 1:
        # L1 prep - same as before
        noisy_reset(circuit, loc1, N_now, p)
        noisy_reset(circuit, loc2, N_now, p)
        detector_0prep = []
    else:
        detector_0prep = []
        
        # Prepare 7 inner L1 blocks (each with its own verification)
        for i in range(n_now):
            detector_0prep.append(append_noisy_0prep_v2(
                circuit, (loc1+i)*N_prev, (loc1+n_now+i)*N_prev, 1, N_prev, p
            ))
        
        # Prepare verification ancilla block at L1
        detector_0prep.append(append_noisy_0prep_v2(
            circuit, (loc2)*N_prev, (loc2+n_now)*N_prev, 1, N_prev, p
        ))
    
    if N_now == N_steane and N_prev != 1:
        # L2 encoding - NO EC ROUNDS!
        
        # Apply H gates to form superposition
        append_h(circuit, (loc1+0)*N_prev, 1, N_prev)
        append_h(circuit, (loc1+1)*N_prev, 1, N_prev)
        append_h(circuit, (loc1+3)*N_prev, 1, N_prev)
        
        # CNOT round 1 (with noise but NO EC)
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+5)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+4)*N_prev, 1, N_prev, p)
        
        # CNOT round 2
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+6)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+4)*N_prev, 1, N_prev, p)
        
        # CNOT round 3
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+5)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc1+6)*N_prev, 1, N_prev, p)
        
        # Verification CNOTs (data -> verification ancilla)
        append_noisy_cnot(circuit, (loc1+2)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+5)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        
        # Measure verification ancilla
        detector_0prep_l2 = append_noisy_m(circuit, (loc2)*N_prev, 1, N_prev, p)
        
        # CRITICAL: Decorrelation CNOTs to uncompute entanglement with measured ancilla
        # This is the key fix - the original was missing this!
        append_cnot(circuit, (loc2)*N_prev, (loc1+2)*N_prev, 1, N_prev)
        append_cnot(circuit, (loc2)*N_prev, (loc1+4)*N_prev, 1, N_prev)
        append_cnot(circuit, (loc2)*N_prev, (loc1+5)*N_prev, 1, N_prev)
        
        # Return detector info - NO detector_X/detector_Z since no EC during prep
        return detector_0prep, detector_0prep_l2, [], []
    
    elif N_now == N_steane:
        # L1 Steane encoding with verification
        append_h(circuit, (loc1+0)*N_prev, 1, N_prev)
        append_h(circuit, (loc1+1)*N_prev, 1, N_prev)
        append_h(circuit, (loc1+3)*N_prev, 1, N_prev)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+5)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+4)*N_prev, 1, N_prev, p)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+6)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+3)*N_prev, (loc1+4)*N_prev, 1, N_prev, p)
        
        append_noisy_cnot(circuit, (loc1+1)*N_prev, (loc1+5)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc1+6)*N_prev, 1, N_prev, p)
        
        # Verification CNOTs
        append_noisy_cnot(circuit, (loc1+2)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+4)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        append_noisy_cnot(circuit, (loc1+5)*N_prev, (loc2)*N_prev, 1, N_prev, p)
        
        # Measure verification
        detector_0prep.append(append_noisy_m(circuit, (loc2)*N_prev, 1, N_prev, p))
        
        # Decorrelation CNOTs
        append_cnot(circuit, (loc2)*N_prev, (loc1+2)*N_prev, 1, N_prev)
        append_cnot(circuit, (loc2)*N_prev, (loc1+4)*N_prev, 1, N_prev)
        append_cnot(circuit, (loc2)*N_prev, (loc1+5)*N_prev, 1, N_prev)
        
        return detector_0prep


def decode_measurement_steane(m, m_type='x'):
    outcome=0
    outcome = np.sum(m[:]*logical_op[:])%2

    syndrome = 0
    for i in range(3):
        e = np.sum(m[:]*check_matrix[i,:])%2
        syndrome += e*(2**i)
    if syndrome > 0:
        outcome = (outcome+logical_op[int(syndrome)-1])%2
    
    return outcome

def decode_measurement_steane_post_selection(m, m_type='x'):
    outcome=0
    outcome = np.sum(m[:]*logical_op[:])%2

    syndrome = 0
    for i in range(3):
        e = np.sum(m[:]*check_matrix[i,:])%2
        syndrome += e*(2**i)
    if syndrome>0:
        return -1
    else:
        return outcome

def append_noisy_ec(circuit, loc1, loc2, loc3, loc4, N_prev, N_now, p):
    global detector_now
    detector_0prep = []
    detector_0prep_l2=[]
    detector_Z=[]
    detector_X=[]
    
    if N_now==1:
        return None
    
    n_now=N_now

    if N_prev==1:
        detector_0prep.extend(append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p))
        detector_0prep.extend(append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p))
    else:
        # Use CORRECTED prep (v2) for L2 ancilla preparation - no EC during prep
        detector_0prep_l1, detector_0prep_l2_tmp, detector_X_l1, detector_Z_l1 = append_noisy_0prep_v2(circuit, loc2, loc4, N_prev, N_now, p)
        detector_0prep.extend(detector_0prep_l1)
        detector_0prep_l2.append(detector_0prep_l2_tmp)
        detector_X.extend(detector_X_l1)  # Will be empty with v2
        detector_Z.extend(detector_Z_l1)  # Will be empty with v2
        detector_0prep_l1, detector_0prep_l2_tmp, detector_X_l1, detector_Z_l1 = append_noisy_0prep_v2(circuit, loc3, loc4, N_prev, N_now, p)
        detector_0prep.extend(detector_0prep_l1)
        detector_0prep_l2.append(detector_0prep_l2_tmp)
        detector_X.extend(detector_X_l1)  # Will be empty with v2
        detector_Z.extend(detector_Z_l1)  # Will be empty with v2
    append_h(circuit,loc2,N_prev,n_now)
    append_noisy_cnot(circuit,loc2,loc3,N_prev,n_now, p)
    
    if N_prev!=1:
        for i in range(n_now):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit,(loc2+i)*N_prev,(loc4+0)*N_prev,(loc4+1)*N_prev,(loc4+2)*N_prev,1,N_prev,p)
            detector_0prep.extend(detector_0prep_c4)
            detector_Z.append(detector_Z_c4)
            detector_X.append(detector_X_c4)
        for i in range(n_now):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit,(loc3+i)*N_prev,(loc4+0)*N_prev,(loc4+1)*N_prev,(loc4+2)*N_prev,1,N_prev,p)
            detector_0prep.extend(detector_0prep_c4)
            detector_Z.append(detector_Z_c4)
            detector_X.append(detector_X_c4)
    
    append_noisy_cnot(circuit,loc1,loc2,N_prev,n_now, p)
    
    append_h(circuit,loc1,N_prev,n_now)
    
    detector_Z.append(append_noisy_m(circuit,loc1,N_prev,n_now, p))
    detector_X.append(append_noisy_m(circuit,loc2,N_prev,n_now, p))
    
    append_swap(circuit,loc1,loc3,N_prev,n_now)
    
    if N_prev==1:
        return detector_0prep, detector_Z, detector_X
    else:
        return detector_0prep, detector_0prep_l2, detector_Z, detector_X

def decode_ec_hd(x, detector_X, detector_Z, correction_x_prev_l1, correction_z_prev_l1):
    """Decode EC round for hierarchical decoder.
    
    Handles both old structure (with prep EC) and new v2 structure (no prep EC).
    - Old structure: 45+45+7+7+1 = 105 entries (num_ec_0prep=45)
    - New structure: 7+7+1 = 15 entries (effective num_ec_0prep=0)
    """
    mx=[[]]*7
    mz=[[]]*7
    correction_x_next_l1=[[]]*7
    correction_z_next_l1=[[]]*7
    cx1=[[]]*7
    cz1=[[]]*7
    cx2=[[]]*7
    cz2=[[]]*7
    cx3=[[]]*7
    cz3=[[]]*7
    
    # Detect structure based on list length
    # Old structure has 90+ entries, new has 15
    is_corrected_prep = len(detector_X) < 50
    
    if is_corrected_prep:
        # NEW v2 structure: no prep EC entries
        # Structure is: [7 L1 EC anc1] + [7 L1 EC anc2] + [transversal meas]
        # So we read from offsets 0, 7, 14 instead of 2*num_ec_0prep, 2*num_ec_0prep+7, 2*num_ec_0prep+14
        for i in range(7):
            cx1[i] = correction_x_prev_l1[i]
            cz1[i] = correction_z_prev_l1[i]
            cx2[i] = decode_measurement_steane(x[detector_X[i][0][0]:detector_X[i][0][1]])
            cz2[i] = decode_measurement_steane(x[detector_Z[i][0][0]:detector_Z[i][0][1]])
            cx3[i] = decode_measurement_steane(x[detector_X[7+i][0][0]:detector_X[7+i][0][1]])
            cz3[i] = decode_measurement_steane(x[detector_Z[7+i][0][0]:detector_Z[7+i][0][1]])
        
        # No prep EC corrections to apply with v2
        
        for i in range(7):
            detx=detector_X[14][i]  # Transversal measurement at index 14
            detz=detector_Z[14][i]
            x_correction = (cx1[i]+cx2[i])%2
            z_correction = (cz1[i]+cz2[i])%2
            correction_x_next_l1[i] = cx3[i]
            correction_z_next_l1[i] = cz3[i]
            mx[i]=(decode_measurement_steane(x[detx[0]:detx[1]], 'x')+x_correction)%2
            mz[i]=(decode_measurement_steane(x[detz[0]:detz[1]], 'z')+z_correction)%2
    else:
        # OLD structure with prep EC entries
        cx2_0prep=[[]]*num_ec_0prep
        cz2_0prep=[[]]*num_ec_0prep
        cx3_0prep=[[]]*num_ec_0prep
        cz3_0prep=[[]]*num_ec_0prep
        
        for i in range(7):
            cx1[i] = correction_x_prev_l1[i]
            cz1[i] = correction_z_prev_l1[i]
            cx2[i] = decode_measurement_steane(x[detector_X[2*num_ec_0prep+i][0][0]:detector_X[2*num_ec_0prep+i][0][1]])
            cz2[i] = decode_measurement_steane(x[detector_Z[2*num_ec_0prep+i][0][0]:detector_Z[2*num_ec_0prep+i][0][1]])
            cx3[i] = decode_measurement_steane(x[detector_X[2*num_ec_0prep+i+7][0][0]:detector_X[2*num_ec_0prep+i+7][0][1]])
            cz3[i] = decode_measurement_steane(x[detector_Z[2*num_ec_0prep+i+7][0][0]:detector_Z[2*num_ec_0prep+i+7][0][1]])

        for a in range(num_ec_0prep):
            cx2_0prep[a]=decode_measurement_steane(x[detector_X[a][0][0]:detector_X[a][0][1]])
            cz2_0prep[a]=decode_measurement_steane(x[detector_Z[a][0][0]:detector_Z[a][0][1]])
            cx3_0prep[a]=decode_measurement_steane(x[detector_X[num_ec_0prep+a][0][0]:detector_X[num_ec_0prep+a][0][1]])
            cz3_0prep[a]=decode_measurement_steane(x[detector_Z[num_ec_0prep+a][0][0]:detector_Z[num_ec_0prep+a][0][1]])
            for i in propagation_l2_0prep_X[a]:
                cz2[i] = (cz2[i]+cx2_0prep[a])%2
                cx3[i] = (cx3[i]+cx3_0prep[a])%2
            for i in propagation_l2_0prep_Z[a]:
                cx2[i] = (cx2[i]+cz2_0prep[a])%2
                cx3[i] = (cx3[i]+cz2_0prep[a])%2
                cz2[i] = (cz2[i]+cz3_0prep[a])%2
                cz3[i] = (cz3[i]+cz3_0prep[a])%2
        
        for i in range(7):
            detx=detector_X[2*num_ec_0prep+14][i]
            detz=detector_Z[2*num_ec_0prep+14][i]
            x_correction = (cx1[i]+cx2[i])%2
            z_correction = (cz1[i]+cz2[i])%2
            correction_x_next_l1[i] = cx3[i]
            correction_z_next_l1[i] = cz3[i]
            mx[i]=(decode_measurement_steane(x[detx[0]:detx[1]], 'x')+x_correction)%2
            mz[i]=(decode_measurement_steane(x[detz[0]:detz[1]], 'z')+z_correction)%2
    
    correction_x = decode_measurement_steane(mx, 'x')
    correction_z = decode_measurement_steane(mz, 'z')
    
    return correction_x, correction_z, correction_x_next_l1, correction_z_next_l1

def decode_m_hd(x, detector_m, correction_l1):
    m=[[]]*7
    for i in range(7):
        det=detector_m[i]
        m[i]=(decode_measurement_steane(x[det[0]:det[1]])+correction_l1[i])%2
    
    outcome = decode_measurement_steane(m, 'x')
    
    return outcome

def post_selection_steane(x,detector_0prep):
    if x[detector_0prep[0]]%2==0:
        return True
    else:
        return False

def post_selection_steane_l2(x,detector_0prep,detector_X,detector_Z):
    outcome = decode_measurement_steane(x[detector_0prep[0]:detector_0prep[1]])
    for a in propagation_l2_0prep_m:
        correction_x = decode_measurement_steane(x[detector_X[a][0][0]:detector_X[a][0][1]])
        outcome = (outcome+correction_x)%2
    if outcome%2==1:
        return False
    else:
        return True

def accept_l1(x):
    global list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ
    
    if no_post_EC==True:
        num_correction=2*Q
    else:
        num_correction=2*(Q+1)
    X_propagate=[[1],[3]]
    Z_propagate=[[0],[2]]
    outcome = np.zeros(4)
    correction_x = np.zeros(num_correction)
    correction_z = np.zeros(num_correction)
    
    for i in range(num_correction):
        correction_x[i] = decode_measurement_steane(x[list_detector_X[i][0][0]:list_detector_X[i][0][1]])
        correction_z[i] = decode_measurement_steane(x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]])
    
    for i in range(4):
        outcome[i] = decode_measurement_steane(x[list_detector_m[i][0]:list_detector_m[i][1]])
        
    for i in range(num_correction):
        pos = i%2
        for x_prop in X_propagate[pos]:
            if outcome[x_prop]==-1:
                continue
            if correction_x[i]==1:
                outcome[x_prop] = (outcome[x_prop]+1)%2
            if correction_x[i]==-1:
                outcome[x_prop] = -1
        for z_prop in Z_propagate[pos]:
            if outcome[z_prop]==-1:
                continue
            if correction_z[i]==1:
                outcome[z_prop] = (outcome[z_prop]+1)%2
            if correction_z[i]==-1:
                outcome[z_prop] = -1
                
    flag=1
    for i in range(4):
        if outcome[i]==1:
            flag=0
        if outcome[i]==-1:
            flag*=0.5
    num_p= 1-flag
    return num_p

def accept_l2(x):
    global list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ
    
    if no_post_EC==True:
        num_correction=2*Q
    else:
        num_correction=2*(Q+1)
    X_propagate=[[1],[3]]
    Z_propagate=[[0],[2]]
    outcome = np.zeros(4)
    correction_x = np.zeros(num_correction)
    correction_z = np.zeros(num_correction)
    correction1_x = np.zeros(7)
    correction1_z = np.zeros(7)
    correction2_x = np.zeros(7)
    correction2_z = np.zeros(7)
    
    for i in range(Q):
        correction_x[2*i], correction_z[2*i], correction1_x, correction1_z = decode_ec_hd(x, list_detector_X[2*i], list_detector_Z[2*i], correction1_x, correction1_z)
        correction_x[2*i+1], correction_z[2*i+1], correction2_x, correction2_z = decode_ec_hd(x, list_detector_X[2*i+1], list_detector_Z[2*i+1], correction2_x, correction2_z)
    
    outcome[0] = decode_m_hd(x, list_detector_m[0],correction1_z)
    outcome[1] = decode_m_hd(x, list_detector_m[1],correction1_x)
    outcome[2] = decode_m_hd(x, list_detector_m[2],correction2_z)
    outcome[3] = decode_m_hd(x, list_detector_m[3],correction2_x)
        
    for i in range(num_correction):
        pos = i%2
        for x_prop in X_propagate[pos]:
            if outcome[x_prop]==-1:
                continue
            if correction_x[i]==1:
                outcome[x_prop] = (outcome[x_prop]+1)%2
            if correction_x[i]==-1:
                outcome[x_prop] = -1
        for z_prop in Z_propagate[pos]:
            if outcome[z_prop]==-1:
                continue
            if correction_z[i]==1:
                outcome[z_prop] = (outcome[z_prop]+1)%2
            if correction_z[i]==-1:
                outcome[z_prop] = -1
                
    flag=1
    for i in range(4):
        if outcome[i]==1:
            flag=0
        if outcome[i]==-1:
            flag*=0.5
    num_p= 1-flag
    return num_p

def post_selection_l1(x):
    global detector_now, list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ,list_detector_0prep
    
    if no_post_EC==True:
        num_correction=2*Q
    else:
        num_correction=2*(Q+1)
        
    for a in list_detector_0prep:
        if len(a)==1:
            if post_selection_steane(x,a[0]) == False:
                return False
        elif len(a)==2:
            if a[1]-a[0]==1:
                if post_selection_steane(x,a) == False:
                    return False
    return True

def post_selection_l2(x):
    global detector_now, list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ,list_detector_0prep,list_detector_0prep_l2
    
    if no_post_EC==True:
        num_correction=2*Q
    else:
        num_correction=2*(Q+1)
        
    for a in list_detector_0prep:
        if len(a)==1:
            if post_selection_steane(x,a[0]) == False:
                return False
        elif len(a)==2:
            if a[1]-a[0]==1:
                if post_selection_steane(x,a) == False:
                    return False
    for i in range(num_correction):
        if post_selection_steane_l2(x,list_detector_0prep_l2[2*i],list_detector_X[i][0:num_ec_0prep],list_detector_Z[i][0:num_ec_0prep])==False:
            return False
        if post_selection_steane_l2(x,list_detector_0prep_l2[2*i+1],list_detector_X[i][num_ec_0prep:2*num_ec_0prep],list_detector_Z[i][num_ec_0prep:2*num_ec_0prep])==False:
            return False
    return True

def post_selection_l2_memory(x, list_detector_0prep, list_detector_0prep_l2, list_detector_X, list_detector_Z, Q, detector_X_prep=None, detector_Z_prep=None):
    """
    Post-selection for single-qubit L2 memory experiment.
    Unlike post_selection_l2 which expects 2 logical qubits (for CNOT),
    this function handles the case of 1 logical qubit with Q rounds of EC.
    
    Args:
        x: Sample vector
        list_detector_0prep: Level-1 preparation detectors
        list_detector_0prep_l2: Level-2 verification detectors [prep_l2, ec_anc1_l2, ec_anc2_l2, ...]
        list_detector_X: EC round X detectors
        list_detector_Z: EC round Z detectors
        Q: Number of EC rounds
        detector_X_prep: X detectors from noisy prep (for prep L2 verification)
        detector_Z_prep: Z detectors from noisy prep (for prep L2 verification)
    """
    # Post-select on level-1 preparation detectors
    for a in list_detector_0prep:
        if len(a)==1:
            if post_selection_steane(x,a[0]) == False:
                return False
        elif len(a)==2:
            if a[1]-a[0]==1:
                if post_selection_steane(x,a) == False:
                    return False
    
    # Post-select on level-2 preparation verification (index 0)
    # This is CRITICAL for fault tolerance - verifies the prep was successful
    # NOTE: With v2 prep, detector_X_prep and detector_Z_prep are empty (no EC during prep)
    # In that case, we only verify the L2 verification measurement, not the EC detectors
    if (len(list_detector_0prep_l2) > 0 and list_detector_0prep_l2[0] is not None):
        # Check if we have prep EC detectors (old-style prep) or not (v2 prep)
        if detector_X_prep is not None and len(detector_X_prep) > 0 and detector_Z_prep is not None and len(detector_Z_prep) > 0:
            if post_selection_steane_l2(x, list_detector_0prep_l2[0], 
                                        detector_X_prep[0:num_ec_0prep], 
                                        detector_Z_prep[0:num_ec_0prep]) == False:
                return False
        else:
            # v2 prep: just verify the L2 measurement is clean (all zeros after decoding)
            if list_detector_0prep_l2[0] is not None:
                detector_l2 = list_detector_0prep_l2[0]
                if isinstance(detector_l2, list) and len(detector_l2) == 2:
                    if decode_measurement_steane(x[detector_l2[0]:detector_l2[1]]) != 0:
                        return False
    
    # Check EC round l2 detectors
    # NOTE: With corrected prep (v2), the EC round's detector_X/detector_Z structure is different:
    # - Old: [45 prep EC anc1] + [45 prep EC anc2] + [7 L1 EC anc1] + [7 L1 EC anc2] + [trans meas]
    # - New: [7 L1 EC anc1] + [7 L1 EC anc2] + [trans meas] (no prep EC entries)
    for i in range(Q):
        # Each EC round adds 2 entries to list_detector_0prep_l2 (one per ancilla block)
        # Index into list_detector_0prep_l2: 1 + 2*i and 1 + 2*i + 1 (after prep at index 0)
        idx_base = 1 + 2*i
        
        # Check if we have old structure (with prep EC) or new structure (no prep EC)
        # Old structure has 90+ entries, new has ~15
        has_prep_ec = len(list_detector_X[i]) > 50 if i < len(list_detector_X) else False
        
        if has_prep_ec:
            # Old-style: use num_ec_0prep indexing
            if idx_base < len(list_detector_0prep_l2):
                if post_selection_steane_l2(x, list_detector_0prep_l2[idx_base], 
                                            list_detector_X[i][0:num_ec_0prep], 
                                            list_detector_Z[i][0:num_ec_0prep]) == False:
                    return False
            if idx_base + 1 < len(list_detector_0prep_l2):
                if post_selection_steane_l2(x, list_detector_0prep_l2[idx_base + 1], 
                                            list_detector_X[i][num_ec_0prep:2*num_ec_0prep], 
                                            list_detector_Z[i][num_ec_0prep:2*num_ec_0prep]) == False:
                    return False
        else:
            # New-style (v2 prep): just check L2 verification measurements
            # With no prep EC, we can only verify the L2 measurement itself
            if idx_base < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base] is not None:
                detector_l2 = list_detector_0prep_l2[idx_base]
                if isinstance(detector_l2, list) and len(detector_l2) == 2:
                    if decode_measurement_steane(x[detector_l2[0]:detector_l2[1]]) != 0:
                        return False
            if idx_base + 1 < len(list_detector_0prep_l2) and list_detector_0prep_l2[idx_base + 1] is not None:
                detector_l2 = list_detector_0prep_l2[idx_base + 1]
                if isinstance(detector_l2, list) and len(detector_l2) == 2:
                    if decode_measurement_steane(x[detector_l2[0]:detector_l2[1]]) != 0:
                        return False
    
    return True

# Estimate the logical error rate of level-1
def estimate_logical_cnot_error_l1(p,num_shots):
    global detector_now,list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ,list_detector_0prep
    N_prev=1
    N_now=N_steane
    n_now=N_now

    NN = 2*N_steane

    Q=10
    no_post_EC=True
    isCZ=False


    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    append_0prep(circuit,0,N_prev,N_now)
    append_0prep(circuit,NN,N_prev,N_now)
    append_0prep(circuit,2*NN,N_prev,N_now)
    append_0prep(circuit,3*NN,N_prev,N_now)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,2*NN,N_prev,n_now)

    append_cnot(circuit,0,NN,N_prev,n_now)
    append_cnot(circuit,2*NN,3*NN,N_prev,n_now)

    #error-free CNOT + noisy CNOT (or CZ)+EC gadget

    for q in range(Q):
        append_cnot(circuit, 0, 2*NN, N_prev,n_now)
        
        append_noisy_cnot(circuit, 0, 2*NN, N_prev,n_now, p)

        detector_0prep, detector_Z, detector_X = append_noisy_ec(circuit,0,4*NN,5*NN,6*NN,N_prev,N_now, p)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

        detector_0prep, detector_Z, detector_X = append_noisy_ec(circuit,2*NN,4*NN,5*NN,6*NN,N_prev,N_now, p)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

    append_cnot(circuit,0,NN,N_prev,n_now)
    append_cnot(circuit,2*NN,3*NN,N_prev,n_now)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,2*NN,N_prev,n_now)

    list_detector_m.append(append_m(circuit,0,N_prev,n_now))
    list_detector_m.append(append_m(circuit,NN,N_prev,n_now))
    list_detector_m.append(append_m(circuit,2*NN,N_prev,n_now))
    list_detector_m.append(append_m(circuit,3*NN,N_prev,n_now))


    sample = circuit.compile_detector_sampler().sample(shots=num_shots)

    sample=[x for x in sample if post_selection_l1(x)]
    num= len(sample)
    err=sum([accept_l1(x) for x in sample])
    print(p,num,err)

    if num>0:
        logical_error=err/(num*Q)
        variance=err/(num*Q)**2
    else:
        logical_error=variance=0

    return logical_error, variance

# Estimate the logical error rate of level-2
def estimate_logical_cnot_error_l2(p,num_shots):
    global detector_now, list_detector_m,list_detector_X,list_detector_Z,Q,no_post_EC,isCZ,list_detector_0prep,list_detector_0prep_l2
    N_prev=N_steane
    N_now=N_steane
    n_now=N_now

    NN = 2*N_steane

    Q=1
    no_post_EC=True
    isCZ=False


    list_detector_0prep=[]
    list_detector_0prep_l2=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    append_0prep(circuit,0,N_prev,N_now)
    append_0prep(circuit,NN,N_prev,N_now)
    append_0prep(circuit,2*NN,N_prev,N_now)
    append_0prep(circuit,3*NN,N_prev,N_now)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,2*NN,N_prev,n_now)

    append_cnot(circuit,0,NN,N_prev,n_now)
    append_cnot(circuit,2*NN,3*NN,N_prev,n_now)

    #error-free CNOT + noisy CNOT (or CZ)+EC gadget

    for q in range(Q):
        append_cnot(circuit, 0, 2*NN, N_prev,n_now)
        
        append_noisy_cnot(circuit, 0, 2*NN, N_prev,n_now, p)

        detector_0prep, detector_0prep_l2, detector_Z, detector_X = append_noisy_ec(circuit,0,4*NN,5*NN,6*NN,N_prev,N_now, p)
        list_detector_0prep.extend(detector_0prep)
        list_detector_0prep_l2.extend(detector_0prep_l2)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

        detector_0prep, detector_0prep_l2, detector_Z, detector_X = append_noisy_ec(circuit,2*NN,4*NN,5*NN,6*NN,N_prev,N_now, p)
        list_detector_0prep.extend(detector_0prep)
        list_detector_0prep_l2.extend(detector_0prep_l2)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

    append_cnot(circuit,0,NN,N_prev,n_now)
    append_cnot(circuit,2*NN,3*NN,N_prev,n_now)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,2*NN,N_prev,n_now)

    list_detector_m.append(append_m(circuit,0,N_prev,n_now))
    list_detector_m.append(append_m(circuit,NN,N_prev,n_now))
    list_detector_m.append(append_m(circuit,2*NN,N_prev,n_now))
    list_detector_m.append(append_m(circuit,3*NN,N_prev,n_now))


    sample = circuit.compile_detector_sampler().sample(shots=num_shots)

    sample=[x for x in sample if post_selection_l2(x)]
    num= len(sample)
    err=sum([accept_l2(x) for x in sample])
    print(p,num,err)

    if num>0:
        logical_error=err/(num*Q)
        variance=err/(num*Q)**2
    else:
        logical_error=variance=0

    return logical_error, variance

if __name__ == '__main__':
    output_file_name = str(sys.argv[1])
    num_shots = int(sys.argv[2])
    case_number = int(sys.argv[3])
    p = float(sys.argv[4])
    error_model = str(sys.argv[5])
    l = int(sys.argv[6])

    if error_model == 'a':
        gamma = p/10
    elif error_model == 'b':
        gamma = p/2
    elif error_model == 'c':
        gamma = p

    N_steane=7

    check_matrix = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]])
    logical_op = np.array([1,1,1,0,0,0,0]) #steane

    # Propagation of Pauli correction of level-1 EC gadget in the level-2 0-state preparation circuit
    propagation_l2_0prep_X =[
        [0,2,4,6],[1,2,5,6],[2],[3,4,5,6],[4,6],[5],
        [0,2],[1,5,6],[2],[3,4,6],[4,6],[5],[6],
        [0],[1,5],[2],[3],[4,6],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]

    propagation_l2_0prep_Z =[
        [0],[1],[2,0,1],[3],[4,3,0],[5,1,3],
        [0],[1],[2,0],[3],[4,3],[5,1],[6,1,4],
        [0],[1],[2],[3],[4],[5,1],[6,4],[2,4,5],
        [0],[1],[2],[3],[4],[5],[6],[4,5],
        [0],[1],[2],[3],[4],[5],[6],[5],
        [0],[1],[2],[3],[4],[5],[6],[]
    ]

    propagation_l2_0prep_m=[2,4,5,6,7,8,9,10,11,14,15,17,18,20,25,26,28,34,36,44]

    num_ec_0prep = 45

    if l!=1 and l!=2:
        print('l should be 1 or 2!')
    else:
        if l==1:
            logical_error, variance = estimate_logical_cnot_error_l1(p,num_shots)
        elif l==2:
            logical_error, variance = estimate_logical_cnot_error_l2(p,num_shots)
        with open(output_file_name, 'a') as file:
            output_data = {
                'physical_error': p,
                'logical_error': logical_error,
                'variance': variance
            }
            file.write('\"case' + str(case_number) + '\":' + json.dumps(output_data) +',\n')

def estimate_memory_logical_error_l2(p, num_shots):
    """
    Estimate the logical error rate for memory in level-2 concatenated Steane code.
    Prepares a logical |0⟩ state, applies 1 round of error correction, then measures.
    """
    global detector_now, list_detector_m, list_detector_X, list_detector_Z, Q, no_post_EC, isCZ, list_detector_0prep, list_detector_0prep_l2
    N_prev = N_steane
    N_now = N_steane
    n_now = N_now
    
    NN = 2 * N_steane
    
    Q = 1  # 1 round of error correction
    no_post_EC = True
    isCZ = False
    
    list_detector_0prep = []
    list_detector_0prep_l2 = []
    list_detector_X = []
    list_detector_Z = []
    list_detector_m = []
    
    detector_now = 0
    
    circuit = stim.Circuit()
    
    # Prepare logical |0⟩ state at level-2 using CORRECTED prep
    # v2 has no EC during prep (reduces noise amplification) and includes decorrelation CNOTs
    # IMPORTANT: detector_X_prep and detector_Z_prep are empty since no EC during prep
    detector_0prep, detector_0prep_l2, detector_X_prep, detector_Z_prep = append_noisy_0prep_v2(circuit, 0, NN, N_prev, N_now, p)
    list_detector_0prep.extend(detector_0prep)
    list_detector_0prep_l2.append(detector_0prep_l2)
    
    # Apply 1 round of error correction
    detector_0prep_ec, detector_0prep_l2_ec, detector_Z_ec, detector_X_ec = append_noisy_ec(circuit, 0, 2*NN, 3*NN, 4*NN, N_prev, N_now, p)
    list_detector_0prep.extend(detector_0prep_ec)
    list_detector_0prep_l2.extend(detector_0prep_l2_ec)
    list_detector_X.append(detector_X_ec)
    list_detector_Z.append(detector_Z_ec)
    
    # Measure the logical state in Z basis
    list_detector_m.append(append_m(circuit, 0, N_prev, n_now))
    
    # Sample from circuit
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)
    
    # Apply post-selection with prep's X/Z detectors for proper L2 verification
    sample = [x for x in sample if post_selection_l2_memory(
        x, list_detector_0prep, list_detector_0prep_l2, list_detector_X, list_detector_Z, Q,
        detector_X_prep=detector_X_prep, detector_Z_prep=detector_Z_prep
    )]
    num = len(sample)
    
    # Count logical errors
    err = 0
    for x in sample:
        # Initialize correction tracking
        correction_x = np.zeros(7)
        correction_z = np.zeros(7)
        outer_x_correction = 0  # Accumulated outer code X correction
        
        # Decode error correction rounds
        # decode_ec_hd returns (outer_x, outer_z, level1_x, level1_z)
        for i in range(Q):
            outer_x, outer_z, correction_x, correction_z = decode_ec_hd(
                x, list_detector_X[i], list_detector_Z[i],
                correction_x, correction_z
            )
            # Accumulate outer code X correction (X errors flip Z-basis measurements)
            if outer_x == 1:
                outer_x_correction = (outer_x_correction + 1) % 2
        
        # Decode final Z-basis measurement with level-1 X corrections
        # X errors cause bit flips in Z-basis measurements!
        outcome = decode_m_hd(x, list_detector_m[0], correction_x)
        
        # Apply outer code X correction to the measurement outcome
        if outer_x_correction == 1:
            outcome = (outcome + 1) % 2
        
        # Expected outcome is 0 (prepared |0⟩ state, measured in Z basis)
        if outcome != 0:
            err += 1
    
    print(p, num, err)
    
    if num > 0:
        logical_error = err / num
        variance = err / num**2
    else:
        logical_error = variance = 0
    
    return logical_error, variance