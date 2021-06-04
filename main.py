import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import cmath
from scipy.sparse.linalg import eigs as spEigs

import utils as ut



class Experiment:
        

    def __init__(self, VOTER_RANGE, NUM_CANDIDATES, NUM_ITERATIONS, NUM_TIES, welfare_vector, WELFARE_TYPE, OUTPUT_FILE_NAME):
        """
        Parameters
        ----------
        VOTER_RANGE : list[int]
            Range from which NUM_VOTERS is selected.
        
        NUM_CANDIDATES : int
            Number of alternatives (3 <= m <= 4).
        
        NUM_ITERATIONS : int
            Number of iterations to run Iterative Plurality.
        
        NUM_TIES : int
            The number of alternatives that should be tied in the potential winning set of the generated profile (-1, or 1 <= NUM_TIES <= NUM_CANDIDATES). If (default = -1), then no ties will be asserted in the generation of the returned profile.
        
        welfare_vector : list[float]
            The utility vector to base the social welfare computations on. By default (welfare_vector=[]) implies the Borda welfare.
        
        WELFARE_TYPE : str
            Name of the utility vector. Used for plotting output.
        
        OUTPUT_FILE_NAME : str
            Where to output important information by running the EADPoA experiment.
        """
        
        self.VOTER_RANGE      = VOTER_RANGE
        self.NUM_CANDIDATES   = NUM_CANDIDATES
        self.NUM_ITERATIONS   = NUM_ITERATIONS
        self.NUM_TIES         = NUM_TIES
        self.welfare_vector   = welfare_vector
        self.WELFARE_TYPE     = WELFARE_TYPE
        self.OUTPUT_FILE_NAME = OUTPUT_FILE_NAME
        
        self.ALL = ut.createAll(NUM_CANDIDATES)


    def run_trial( self ):
        """
        Runs a single trial of Iterative Plurality Voting. 
        
        Generates a uniformly random profile P, determines the size of PW(P), computes the adversarial loss D+(P), and outputs other meaningful information. Currently built for only NUM_CANDIDATES = {3,4}.
        
        Returns
        -------
        float
            Adversarial loss for the randomly generated truthful profile.
            
        OUTPUT_STRING : list[str]
            List of important information to output for the trial. Currently, the list is [b_star,NUM_TIES,EQUILIBRIUM_SET], where EQUILIBRIUM_SET (EW(P)) are the equilibrium winning alternatives for P. 
        
        """
        
        # Get truthful profile and potential winning set PW(P)
        b_star, PotWin = ut.get_b_star( self.ALL, self.NUM_VOTERS, self.NUM_CANDIDATES, self.NUM_TIES )
        OUTPUT_STRING = [
            ut.filter_b_star( self.ALL, b_star ),
            str(len(PotWin)), 
            #ut.get_SW_string(b_star),
        ]
        
        
        if len(PotWin) == 1:
            # Nash Equilibrium profile with no BR dynamics
            
            OUTPUT_STRING.append(  str(ut.get_victor(ut.get_truth(b_star))) )
            return 0, OUTPUT_STRING
        elif len(PotWin) == 2:
            # Applies (Lemma 1) to determine which alternative is the unique equilibrium winner. [EQUILIBRIUM] is of length (n) in order to compute the social welfare correctly.
            
            EQUILIBRIUM = [str(ut.check_Tab_Tba(b_star, PotWin))*self.NUM_VOTERS]
            OUTPUT_STRING.append( ';'.join([eq[0] for eq in EQUILIBRIUM]) )
        elif len(PotWin) == 3:
            # Determines which alternatives are in the equilibrium winning set. See ut.equil_three_way(). [EQUILIBRIUM] is of length (n) in order to compute the social welfare correctly.
        
            EQUILIBRIUM = ut.equil_three_way(b_star, PotWin, NUM_CANDIDATES)
            EQUILIBRIUM = [str(x)*self.NUM_VOTERS for x in EQUILIBRIUM]
            OUTPUT_STRING.append( ';'.join([eq[0] for eq in EQUILIBRIUM]) )
        elif len(PotWin) == 4:
            # Determines which alternatives are in the equilibrium set, assuming certain conditions on the input profile (b_star). See ut.run_four_analysis().
            
            WELF_DIFF, EW_SET = ut.run_four_analysis( b_star )
            OUTPUT_STRING.append( EW_SET )
            return WELF_DIFF, OUTPUT_STRING
        else:
            raise NotImplementedError
        
        
        # Computes Adversarial Loss given EW(P) for |PW(P)| = {2,3}
        TRUTH_WELFARE = ut.get_welfare(ut.get_truth(b_star), b_star, NUM_CANDIDATES, welfare_vector)
        EQLZ = [ut.get_welfare(eq, b_star, NUM_CANDIDATES, welfare_vector) for eq in EQUILIBRIUM]
        MIN_EQ_WELFARE = min(EQLZ)
        return TRUTH_WELFARE-MIN_EQ_WELFARE, OUTPUT_STRING
    # end run_trial()


    def run_EADPoA(self, shouldPrintln=True, shouldOutput=True, shouldPlot=True):
        """
        Runs full experiment for Iterative Plurality Voting. 
        
        Runs trials by iterating over a desired range of agents (self.VOTER_RANGE) and number of iterations (self.NUM_ITERATIONS).
        
        Parameters
        ----------
        shouldPrintln : bool, optional
            Whether the program should print out data to stdout while running. (default=True)
        
        shouldOutput : bool, optional
            Whether the program should store meaningful information in a file, once running is completed. (default=True)
        
        shouldPlot : bool, optional
            Whether the program should plot mean and (95%) confidence intervals, once running is completed. (default=True)
        
        Returns
        -------
        float
            Adversarial loss for the randomly generated truthful profile.
            
        OUTPUT_STRING : list[str]
            List of important information to output for the trial. Currently, the list is [b_star,NUM_TIES,EQUILIBRIUM_SET], where EQUILIBRIUM_SET (EW(P)) are the equilibrium winning alternatives for P. 
        
        """
        
        if shouldOutput:
            with open(self.OUTPUT_FILE_NAME, "w") as f:
                f.write("b_star,NUM_TIES,r(PT),D+borda\n")
        
        # Data to collect -- overall
        ADPoA_means = {}
        ADPoA_confs = {}
        COUNT_NUM_4_TIE_FAILS = []
        start_time = time.time()
        
        print("Running Experiment...")
        # Outer Loop: iterate over desired number of voters
        for NUM_VOTERS in self.VOTER_RANGE:
            if shouldPrintln:
                print("\n=========================================\n\nRUNNING NUM_VOTERS = ", NUM_VOTERS, ".......\n=========================================\n")

            # Data to collect -- per NUM_VOTERS
            OUTPUT_DATA = []
            ADPoA_counts = []
            ADPoA_means[ NUM_VOTERS ] = []
            ADPoA_confs[ NUM_VOTERS ] = []
            
            # Inner Loop: iterate over iterations
            for num_iter in range(self.NUM_ITERATIONS):
                if (num_iter % 10000 == 0 and shouldPrintln): #10K
                    print("\t... i = ", num_iter)
                    
                if (num_iter % 100000 == 0 and shouldPrintln): #100K
                    print("Time: ", round(time.time() - start_time, 5) )
                    print("===========\n")
                
                # Run trial
                outcome, OUTPUT_STRING = self.run_trial()
                
                # Interpret data into appropriate data structures
                if OUTPUT_STRING[-1][0] == "*":
                    COUNT_NUM_4_TIE_FAILS.append( OUTPUT_STRING[-1][1:] )
                OUTPUT_STRING.append(str(outcome))
                OUTPUT_DATA.append(OUTPUT_STRING)
                ADPoA_counts.append(outcome)
            # end inner loop
                
            # Interpret data into appropriate data structures
            AVGs, CI = ut.CONF( [ADPoA_counts] )
            ADPoA_means[ NUM_VOTERS ] = AVGs[0]
            ADPoA_confs[ NUM_VOTERS ] = CI[0]
            
            if shouldPrintln:
                print("\n===========")
                print("MEANs: ", ADPoA_means)
                print("\nCIs: ", ADPoA_confs)
                print("\nTime: ", round(time.time() - start_time,5) )
                print("\nList 4-tie fails: ", COUNT_NUM_4_TIE_FAILS)
                print("===========\n")
            
            if shouldOutput:
                with open(self.OUTPUT_FILE_NAME, "a") as f:
                    for text in OUTPUT_DATA:
                        f.write(','.join(text))
                        f.write('\n')
        # end outer loop
                
        print("\n\n==========================\n\n\nExperiment Complete!")
        
        if shouldPlot:
            print("\nPlotting...")
            plt.errorbar(self.VOTER_RANGE, list(ADPoA_means.values()), yerr=list(ADPoA_confs.values()), fmt='-o')
            plt.xlabel("Num Agents")
            plt.ylabel("Adversarial Loss")
            plt.title(f"m = {NUM_CANDIDATES}; Num Iter = {NUM_ITERATIONS}; {self.WELFARE_TYPE} welfare")
            #plt.xticks([0] + VOTER_RANGE[1::2], [0] + VOTER_RANGE[1::2])            
            plt.show()
    # end run_EADPoA()
# end Experiment()


if __name__ == "__main__":
    
    VOTER_RANGE = list(range(100,1100,100))
    NUM_CANDIDATES = 4
    NUM_ITERATIONS = int(2.5*10**6) #100000 # 100K
    NUM_TIES = -1
    welfare_vector = list(range(NUM_CANDIDATES-1,-1,-1))
    OUTPUT_FILE_NAME = "XXX.txt"
        
    exp = Experiment( VOTER_RANGE, NUM_CANDIDATES, NUM_ITERATIONS, NUM_TIES, welfare_vector, "Borda", OUTPUT_FILE_NAME )
    
    exp.run_EADPoA()











    













