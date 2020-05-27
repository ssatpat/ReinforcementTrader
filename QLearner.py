
import numpy as np  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import random as rand  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
class QLearner(object):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        self.verbose = verbose  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        self.num_actions = num_actions  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        self.s = 0  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar =rar
        self.radr = radr
        self.dyna = dyna
        np.random.seed(314)
        self.Q_table = np.random.rand(num_states, num_actions)
        self.experiences = []
        self.Tc = np.ones((num_states,num_actions))
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    def querysetstate(self, s):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @summary: Update the state without updating the Q-table  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @param s: The new state  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @returns: The selected action  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        self.s = s  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        # action = rand.randint(0, self.num_actions-1)
        if rand.uniform(0.0, 1.0) <= self.rar:  # going rogue
            action = rand.randint(0, 2)
        else:
            action = np.argmax(self.Q_table[s, :])
        self.a = action

        if self.verbose: print(f"s = {s}, a = {action}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        return action  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    def query(self,s_prime,r):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @summary: Update the Q table and return an action  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @param s_prime: The new state  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @param r: The reward  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        @returns: The selected action  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        action =  np.argmax(self.Q_table[s_prime, :])
        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")

        self.updateQ_table(self.s, self.a, s_prime, r)

        self.experiences.append((self.s, self.a, s_prime, r))
        self.rar *= self.radr
        if self.dyna > 0 :
            #now we hallucinate experiences
            #pick a random experience
            for i in range(0, self.dyna):
                experience = rand.choice(self.experiences)
                s = experience[0]
                a = experience[1]
                sprime = experience[2]
                r = experience[3]

                self.updateQ_table(s, a, sprime, r)


        self.s = s_prime
        self.a = action

        return action

    def updateQ_table(self, s, a, s_prime, r):
        self.Q_table[s, a] = (1- self.alpha)*self.Q_table[s, a] + \
                            self.alpha*(r + self.gamma*self.Q_table[s_prime, np.argmax(self.Q_table[s_prime])])

    def author(self):
        return 'ssatpathy9'
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
if __name__=="__main__":  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
