import numpy as np 
import random

# Hopping function--------------------------------------------------------------
def hopping(pos, case, bias_factor):

  """
  Simulates a random walk (hop) with stochastic resetting. 

  Inputs:

   pos:                   current position of the particle
   case:                  unbiased_hop/forward_biased_hop/backward_biased_hop
   bias_factor:           difference in probabilities associated with forward 
                          and backward hopping
  Returns:

    pos:                  position of the particle after one hop has commenced

  """

  w_1 = 50 + bias_factor
  w_2 = 50 - bias_factor

  if str(case) == 'forward_biased_hop':

    # initializes 1-D array that consists of -1s and 1s. A forward bias entails 
    # more -1s and a higher chance of hopping towards the recovered state.
    dpos = random.choices([-1, 1], weights = [w_1, w_2], k=1)

  elif str(case) == 'backward_biased_hop':

    # initializes 1-D array that consists of -1s and 1s. A backward bias entails 
    # more +1s and a higher chance of hopping away from the recovered state.
    dpos = random.choices([-1, 1], weights = [w_2, w_1], k=1)

  else:

    # initializes 1-D array that consists of -1s and 1s. No bias entails 
    # an equal number of + and - 1s and equal chances of forward and 
    # backward hopping.
    dpos = random.choices([-1, 1], weights = [50, 50], k=1)
    
  pos += dpos[0]

  return pos

# Resetting function ------------------------------------------------------------

def resetting(pos, resetting_probability):

  """
  Simulates the choice of resetting. 

  Inputs:

   pos:                   current position of the particle
   case:                  unbiased_hop/forward_biased_hop/backward_biased_hop
   resetting_probability:  probability of resetting when particle is within 
                          resetting region
  Returns:

    pos:                  position of the particle after the choice of whether
                          or not to reset has commenced

  """
  p_c = resetting_probability*100

  # initializes 1-D array that consists of -pos+1 and 0, which results in either
  # pos = -1 (recovery via resetting) or in no change in pos. 
  dpos = random.choices([-(pos+1), 0], weights = [p_c, 100-p_c], k=1)

  pos += dpos[0]
  return pos

  # Note: While position of recovery is technically zero, we momentarily set it
  # to -1 to distinguish between trajectories that recover via hopping and those
  # that do so via resetting.

# Trajectories -----------------------------------------------------------------

def return_trajectories(case, bias_factor, resetting_probability, start, 
                        steps, reset_region, mcs, filename):

  """
  Simulates particle trajectories for different cases of random walk and 
  resetting. 

  Inputs:

   case:                  unbiased_hop/forward_biased_hop/backward_biased_hop
   bias_factor:           difference in probabilities associated with forward 
                          and backward hopping
   resetting_probability:  probability of resetting when particle is within 
                          resetting region
   start:                 initial position of the particle
   steps:                 maximum number of steps of random walk/resetting
   reset_region:          maximum position from which resetting can occur
   mcs:                   number of Monte Carlo simulations

  Returns:

    trajectories:         array containing the position of the particle at
                          each timepoint for each Monte Carlo simulation

  """

  # Initializes data matrices
  trajectories = np.zeros([mcs, steps+1], int)

  # Performs indicated number of Monte Carlo Simulations 
  for j in range(mcs):

    # Sets position of particle to indicated starting point
    pos = start
    trajectories[j,0] = pos
    
    
    # Performs random walk until indicated number of steps
    i = 1
    while i < steps+1:

      # If within resetting region, position is either set to 0 or unchanged.
      if pos <= reset_region:
        pos = resetting(pos, resetting_probability)
        trajectories[j,i] = pos

        # If position resets, perform a new mcs. 
        if pos <= 0:
          break

        # If position is unchanged, proceed with hopping function.
        else: pass

      # If position is beyond resetting region, proceed with hopping function.
      else: pass

      # Position is either moved 1 step to the right or 1 step to the left.
      pos = hopping(pos, case, bias_factor)
      trajectories[j,i] = pos

      # If position reaches 0, perform a new mcs.
      if pos <= 0:
          i += 1
          break

      # If position is not 0, continue with random walk 
      else: pass
      i+= 1

  # Records trajectories

  Filepath = "" 
  + str(filename) + str(".csv")
  
  np.savetxt(Filepath, trajectories, delimiter = ",")

  return trajectories

# Recovery times ---------------------------------------------------------------

def return_recovery_time(single_traj, reset_duration, h):

  """
  Computes recovery time given a single trajectory.

  Inputs:

   single_traj:           1xN array containing a single particle trajectory
                          within the observation time (i.e., each row of 
                          the output of the function, return_trajectories())
   reset_duration:        duration of each instance of resetting
   h:                     duration of each hop

  Returns:

    recovery time:        no. of time steps until particle reaches pos = 0
                          

  """

  recovery_time = 0

  # calculate recovery time
  for i in range(1, len(single_traj)):
    
    if single_traj[i] == -1: 
      recovery_time = (i-1)/h + reset_duration
      break
      
    elif single_traj[i] == 0:
      recovery_time = i/h
      break
      
    else: 

      pass

  return recovery_time

# Mean recovery times ----------------------------------------------------------

def return_mrt(case, bias_factor, resetting_probability, start, steps, 
               reset_region, mcs, reset_duration, h):

  """
  Simulates particle trajectories for different cases of random walk and 
  resetting. Calculates and returns conditional mean recovery time over
  indicated number of Monte Carlo Simulations.

  Inputs:

   case:                  unbiased_hop/forward_biased_hop/backward_biased_hop
   bias_factor:           difference in probabilities associated with forward 
                          and backward hopping
   resetting_probability:  probability of resetting when particle is within 
                          resetting region
   start:                 initial position of the particle
   steps:                 maximum number of steps of random walk/resetting
   reset_region:          maximum position from which resetting can occur
   mcs_steps:             number of Monte Carlo simulations

  Returns:

    conditional mean 
    recovery time:        average recovery time (given recovery) over 
                          indicated number of Monte Carlo simulations 
                          

  """

  recovery_time = np.zeros(mcs)

  # Performs indicated number of Monte Carlo Simulations 
  for j in range(mcs):

    # Sets position of particle to indicated starting point
    pos = start
    
    # Performs random walk until indicated number of steps
    i = 1
    while i < steps+1:

      # If within resetting region, position is either set to 0 or unchanged.
      if pos <= reset_region:
        pos = resetting(pos, resetting_probability)
        
        # If position resets, perform a new mcs. 
        if pos <= 0:
          recovery_time[j] = (i-1)/h + reset_duration
          break

        # If position is unchanged, proceed with hopping function.
        else: pass

      # If position is beyond resetting region, proceed with hopping function.
      else: pass

      # Position is either moved 1 step to the right or 1 step to the left.
      pos = hopping(pos, case, bias_factor)

      # If position reaches 0, perform a new mcs.
      if pos <= 0:
          i += 1
          recovery_time[j] = i/h
          break

      # If position is not 0, continue with random walk 
      else: pass
      i+= 1

  # ----------------------------------------------------------------------
  conditional_mrt = np.mean(recovery_time[recovery_time !=0 ])

  return conditional_mrt