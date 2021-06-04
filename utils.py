import numpy as np
import networkx as nx
import itertools
import copy
import scipy
import scipy.sparse
import scipy.stats as st



# ----------------------------------------------
# -------------- HELPER FUNCTIONS --------------
# ----------------------------------------------


def get_b_star( ALL, NUM_VOTERS, NUM_CANDIDATES, NUM_TIES=-1 ):
    """ Returns a uniformly random generated truthful profile for NUM_VOTERS agents among the NUM_CANDIDATES alternatives.
    
    Parameters
    ----------
    ALL : list[str]
        List of all possible rankings (strings) to generate profile from. Rankings are zero-index and of the format '21340' == '2>1>3>4>0'.
    
    NUM_VOTERS : int
        Number of agents (n > 0).
    
    NUM_CANDIDATES : int
        Number of alternatives (3 <= m <= 10).
    
    NUM_TIES : int, optional
        The number of alternatives that should be tied in the potential winning set of the generated profile (-1, or 1 <= NUM_TIES <= NUM_CANDIDATES). If (-1), then no ties will be asserted in the generation of the returned profile.
        
    Returns
    -------
    b_star : list[str]
        List of rankings
        
    PotWin : str
        Potential Winning Set according to the generated profile
    
    """

    if NUM_TIES != -1:
        # Assert NUM_TIES-way ties:
        PotWin = []
        while len(PotWin) != NUM_TIES:
            b_star = list(np.random.choice(ALL, NUM_VOTERS, replace=1))
            PotWin = get_pot_W_t(get_truth(b_star), NUM_CANDIDATES)
    else:
        b_star = list(np.random.choice(ALL, NUM_VOTERS, replace=1))
        PotWin = get_pot_W_t(get_truth(b_star), NUM_CANDIDATES)

    return b_star, PotWin


def createAll(NUM_CANDIDATES):
    """
    Creates all (m!) possible rankings for the NUM_CANDIDATES alternatives.
    
    Parameters
    ----------
    NUM_CANDIDATES : int
        Number of alternatives (3 <= m <= 10).
        
    Returns
    -------
    list[str]
        Returns list of (m!) unique rankings. Rankings are zero-index and of the format '21340' == '2>1>3>4>0'.
    
    """
    
    ALL = []
    ALL = list(itertools.permutations(range(NUM_CANDIDATES)))
    ALL = [''.join(map(str,k)) for k in ALL]
    return ALL


def get_truth(b_star):
    """
    Returns top preferences for each voter
    
    Parameters
    ----------
    b_star : list[str]
        Profile (not necessarily truthful) of rankings.
    
    Returns
    -------
    str
        Returns a 'State' of top preferences for each agent, according to the input profile.
    
    """
    
    return ''.join([k[0] for k in b_star])


def get_victor(state): 
    """
    Returns the plurality winning alternative according to the input State, subject to alphabetical tie-breaking.
    
    Parameters
    ----------
    state : str
        List of alternatives
    
    Returns
    -------
    char
        Returns the plurality winning alternative according to the input State. TYPE: str(int(c)), for (0 <= c <= NUM_CANDIDATES-1).
    
    """
    
    victor = np.unique(list(state), return_counts=1)
    return victor[0][np.argmax(victor[1])]
        

def get_score(state, NUM_CANDIDATES):
    """
    Returns the plurality score for each alternative in [0,1,...,NUM_CANDIDATES-1] according to state
    
    Parameters
    ----------
    state : str
        List of alternatives.
    
    NUM_CANDIDATES : int
        Number of alternatives (3 <= m <= 10).
    
    Returns
    -------
    list[int]
        Returns the plurality score for each alternative in [0,1,...,NUM_CANDIDATES-1] according to state
    
    """
    return [state.count(str(j)) for j in range(NUM_CANDIDATES)]


def CONF( data ):
    """
    Returns 95% confidence intervals surrounding the mean of the data. Data is interpreted as a list of lists of floats. If there is only one input X = [...], use 'CONF([X])'.
    
    Parameters
    ----------
    data : list[ list[float] ]
        List of lists of floats. Will return one AVG and CI for each list in data.
    
    Returns
    -------
    AVGs : list[float]
        One average for each list in data
    
    CI : list[float]
        One 95% confidence interval for each list in data
    
    """
    
    AVGs = [np.mean(x) for x in data]
    STDs = [np.std(x, ddof=1) for x in data]
    ENN =  [len(x) for x in data]
    DDOF = [st.t.ppf(1-0.025, len(x)-1) for x in data] # t-scores for 95% CI
    CI = [DDOF[i]*STDs[i]/np.sqrt(ENN[i]) for i in range(len(data))]
    return AVGs, CI


def filter_b_star( ALL, b_star ):
    """
    Restates the profile b_star in terms of its histogram.
    
    Parameters
    ----------
    ALL : list[str]
        List of (m!) unique rankings.
    
    b_star : list[str]
        Profile (not necessarily truthful) of rankings.
    
    Returns
    -------
    str
        Ex. (m=3) "1;2;0;3;1;2" corresponds to (1) of ALL[0], (2) of ALL[1], etc.
    
    """
    return ';'.join([str(b_star.count(k)) for k in ALL])


def get_pot_W_t(state, NUM_CANDIDATES):
    """
    Returns the potential winning set at 'state'. 
    
    Recall that 
    
    PW(P) = { score_P(a) = score_P( r(P) )-1 if a is before r(P)
                    score_P(a) = score_P( r(P) )   if a is after  r(P) }
    
    Parameters
    ----------
    state : str
        List of alternatives.
    
    NUM_CANDIDATES : int
        Number of alternatives (3 <= m <= 10).
    
    Returns
    -------
    list[int]
        List of integers representing which alternatives are in the potential winning set of the given state.
    """
    
    S_elt = get_score(state, NUM_CANDIDATES)
    max_elt = max(S_elt)
    curr_winner = S_elt.index(max_elt)
    PotWin = list(filter(lambda i : (S_elt[i] == max_elt-1 and i < curr_winner) or (S_elt[i] == max_elt and i >= curr_winner), range(NUM_CANDIDATES)))
    return PotWin
 






    
# ----------------------------------------------
# ----------- MAIN RUNNER FUNCTIONS ------------
# ----------------------------------------------



def check_Tab_Tba(b_star, W_0): #W_0 = (a,b)
    # O(1) TIME ALGORITHM
    state = get_truth(b_star)
    Tab = 0
    Tba = 0
    #print("\n\nSTATE: ", state, "W_0: ", W_0)
    for i in range(len(state)):
        if not int(state[i]) in W_0:
            #print(">> checking b_star[i] = ", b_star[i], "...")
            #if int(get_pref(b_star[i], W_0[0])) > int(get_pref(b_star[i], W_0[1])): # <-- deprecated line
            if int(b_star[i].index(str(W_0[0]))) < int(b_star[i].index(str(W_0[1]))):
            #    print("TAB")
                Tab += 1
            else:
            #    print("TBA")
                Tba += 1
    
    num_a_votes = list(state).count(str(W_0[0]))
    num_b_votes = list(state).count(str(W_0[1]))
    #print(">>>CHECKING T_AB vs T_BA: \n(Tab, #a), (Tba, #b) = ", (Tab, num_a_votes), (Tba, num_b_votes))
    
    if num_a_votes + Tab >= num_b_votes + Tba:
        return W_0[0]
    else:
        return W_0[1]


def equil_three_way(b_star, W_0, NUM_CANDIDATES): #W_0 = (a,b,c)

    equil_candidates = []
    by_default = True
    
    out_ab = check_Tab_Tba(b_star, [W_0[0], W_0[1]])
    out_ac = check_Tab_Tba(b_star, [W_0[0], W_0[2]])
    out_bc = check_Tab_Tba(b_star, [W_0[1], W_0[2]])
    
    A, Au = pref_existence(b_star, W_0)
    stage = np.argmax(np.array(get_score(get_truth(b_star), NUM_CANDIDATES))[W_0])
    
    if stage == 0:
        if A[2][1] or Au[1]: # (A1: compare ab)
            by_default = 0
            equil_candidates.append(out_ab)
        if A[1][2] or Au[2]: # (A2: compare ac)
            by_default = 0
            equil_candidates.append(out_ac)
        if Au[2] and (Au[1] or A[0][1]): # (A3: compare bc)
            by_default = 0
            equil_candidates.append(out_bc)
        if by_default: # (A4: default to 'a')
            equil_candidates.append(W_0[0])
    elif stage == 2:
        if A[1][0] or Au[0]: # (A1: compare ac)
            by_default = 0
            equil_candidates.append(out_ac)
        if A[0][1] or Au[1]: # (A2: compare bc)
            by_default = 0
            equil_candidates.append(out_bc)
        if Au[1] and (Au[0] or A[2][0]): # (A3: compare ab)
            by_default = 0
            equil_candidates.append(out_ab)
        if by_default: # (A4: default to 'c')
            equil_candidates.append(W_0[2])
    elif stage == 1:
        if A[0][2] or Au[2]: # (A1: compare bc)
            by_default = 0
            equil_candidates.append(out_bc)
        if A[2][0] or Au[0]: # (A2: compare ab)
            by_default = 0
            equil_candidates.append(out_ab)
        if Au[0] and (Au[2] or A[1][2]): # (A3: compare ac)
            by_default = 0
            equil_candidates.append(out_ac)
        if by_default: # (A4: default to 'b')
            equil_candidates.append(W_0[1])
    
    return np.unique(equil_candidates)





def run_four_analysis( b_star ):
    # Using Borda welfare.

    if not contains_all( b_star ):
        return 0, '*'

    check_01 = check_Tab_Tba( b_star, [0,1] )
    check_02 = check_Tab_Tba( b_star, [0,2] )
    check_03 = check_Tab_Tba( b_star, [0,3] )
    check_12 = check_Tab_Tba( b_star, [1,2] )
    check_13 = check_Tab_Tba( b_star, [1,3] )
    check_23 = check_Tab_Tba( b_star, [2,3] )
    TO_CHECK = list(set([check_01, check_02, check_03, check_12, check_13, check_23]))
    
    #TO_CHECK = [0,1,2,3]
    EW_SET = ';'.join([str(k) for k in TO_CHECK])
    
    WELF = [get_welfare( str(i), b_star, 4) for i in TO_CHECK]
    return get_welfare( get_victor(get_truth(b_star)), b_star, 4) - min(WELF), EW_SET
        



# ----------------------------------------------
# ----------- OTHER HELPER FUNCTIONS -----------
# ----------------------------------------------


def contains_all( b_star ):
    ALL = ['01','02','03','10','12','13','20','21','23','30','31','32']
    adj = [k[:2] for k in b_star]
    count = sum([k in adj for k in ALL])
    return count == len(ALL)


def pref_existence(b_star, W_0):
    # returns a binary 3x3x2 matrix depicting relevant transition rankings
    # A[0,i,j]=1 depicts the existence of some ranking such that a voter for i in W_0 prefers j in W_0 next
    # A[1,i,j]=1 depicts the existence of some ranking such that a non-pivotal voter prefers i, then j in W_0
    
    # call with "utils.get_pot_W_t(get_truth(b_star), NUM_CANDIDATES)"
    #print("B_STAR W_0 EXTRACTED: ", [extract_potential(b, W_0) for b in b_star])
    
    A = np.zeros((3,3))
    Au = np.zeros(3)
    for ranking_b in b_star:
        ranking_adjusted = extract_potential(ranking_b, W_0)
        if int(int(ranking_b[0]) in W_0): # pivotal voter
            A[W_0.index(int(ranking_adjusted[0])), W_0.index(int(ranking_adjusted[1]))] = 1
        else:
            Au[W_0.index(int(ranking_adjusted[0]))] = 1
    return A, Au


def extract_potential(ranking_b, W_0):
    # transforms a ranking to only have candidates in W_0
    # eg. f('21043', [0,2,4]) = '204'
    return ''.join(list(map(lambda x: x if int(x) in W_0 else '', ranking_b)))


# ----------------------------------------------
# -------------- WELFARE FUNCTIONS -------------
# ----------------------------------------------



 
def get_welfare(state, b_star, NUM_CANDIDATES, welfare_vector = []):
    #print("GETTING WELFARE...: ", (state, b_star))
    victor = get_victor(state)
    #print("VICTOR: ", victor)
    
    if welfare_vector == []: # default to Borda
        welfare_vector = list(range(NUM_CANDIDATES-1,-1,-1)) #[NUM_CANDIDATES-i for i in range(NUM_CANDIDATES)]
    elif welfare_vector == "plu": # get score
        welfare_vector = [1] + [0]*(NUM_CANDIDATES-1)
    
    
    #print(">>> welfare_vector: ", welfare_vector)
    
    welfare = 0
    for i in range(len(b_star)):
        #print("\nchecking (", b_star[i], ")...")
        #x = -1*int(get_pref(b_star[i], victor))
        #print("x: ", x)
        #x = welfare_vector[x]
        #x = welfare_vector[-1*int(get_pref(b_star[i], victor))] # <-- deprecated line
        x = welfare_vector[list(b_star[i]).index(str(victor))]
        #print("welfare[x]: ", x)
        welfare += x
        
    return welfare
    


def get_SW_string( b_star ):
    # Returns social welfare of each of 4 candidates in the form ['a_b_c_d', 'a_b_c_d', 'a_b_c_d', 'a_b_c_d'] where the index in array is for candidates 0-3 and each character is the multiplicity of each position. ex. '2_0_1_0' => SW(a) = 2(u_1) + 1(u_3).
    P = [[j[k] for j in b_star] for k in range(4)]
    return ';'.join(['_'.join([str(j.count(k)) for j in P]) for k in ['0','1','2','3']])

def get_borda_welf( b_star ):
    """
    Assumes NUM_CANDIDATES = 4.
    """
    return ';'.join([str(get_welfare([k], b_star, 4)) for k in ['0','1','2','3']])


# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------

    
    





