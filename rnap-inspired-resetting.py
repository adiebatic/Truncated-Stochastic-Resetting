import numpy as np 
import random


# =============================================================================
# Hopping function
# =============================================================================
def hopping(pos, case, bias_factor):

  """
  Simulates a single step (hop) of a 1-D random walk with optional bias.

  Inputs:
   pos:          current position of the particle
   case:         type of hopping rule
                 ('unbiased_hop', 'forward_biased_hop', 'backward_biased_hop')
   bias_factor:  controls the difference in probabilities between forward
                 and backward hopping

  Returns:
   pos:          updated position of the particle after the hop
  """

  # Define weights controlling hopping probabilities.
  # The base probability is 50/50; bias_factor shifts probability weight.
  w_1 = 50 + bias_factor
  w_2 = 50 - bias_factor

  if str(case) == 'forward_biased_hop':

    # Creates a stochastic step chosen from {-1, +1}.
    # A forward bias means higher probability for -1,
    # which moves the particle toward the recovered state (position 0).
    dpos = random.choices([-1, 1], weights = [w_1, w_2], k=1)

  elif str(case) == 'backward_biased_hop':

    # Backward bias increases probability of +1,
    # meaning the particle is more likely to move away from recovery.
    dpos = random.choices([-1, 1], weights = [w_2, w_1], k=1)

  else:

    # Unbiased hopping: equal probability of stepping left or right.
    dpos = random.choices([-1, 1], weights = [50, 50], k=1)
    
  # Update particle position
  pos += dpos[0]

  return pos


# =============================================================================
# Resetting function
# =============================================================================
def resetting(pos, resetting_probability):

  """
  Simulates the stochastic resetting decision.

  When the particle is inside the resetting region, it can either:
   • reset to the recovery state
   • remain in its current position

  Inputs:
   pos:                    current particle position
   resetting_probability:  probability of resetting when in resetting region

  Returns:
   pos:                    updated particle position
  """

  # Convert probability into percentage weight for random.choices()
  p_c = resetting_probability*100

  # Random choice:
  # -(pos+1) moves the particle to position -1 (reset event)
  # 0 means the particle remains where it is
  dpos = random.choices([-(pos+1), 0], weights = [p_c, 100-p_c], k=1)

  # Apply change in position
  pos += dpos[0]

  return pos

  # NOTE:
  # Recovery position is technically 0.
  # However, resetting sets position to -1 temporarily so that the
  # simulation can distinguish between:
  #   - recovery via hopping (pos = 0)
  #   - recovery via resetting (pos = -1)


# =============================================================================
# Trajectory simulation
# =============================================================================
def return_trajectories(case, bias_factor, resetting_probability, start, 
                        steps, reset_region, mcs, filename):

  """
  Simulates multiple particle trajectories using Monte Carlo simulations.

  Each trajectory represents a random walk with possible resetting.

  Inputs:
   case:                  hopping rule
   bias_factor:           controls hopping bias
   resetting_probability: probability of resetting in reset region
   start:                 initial particle position
   steps:                 maximum number of simulation steps
   reset_region:          maximum position where resetting can occur
   mcs:                   number of Monte Carlo simulations
   filename:              file name to save trajectories

  Returns:
   trajectories:          matrix containing particle position at each time
                          step for each Monte Carlo simulation
  """

  # Initialize matrix storing trajectories
  # Rows = Monte Carlo runs
  # Columns = time steps
  trajectories = np.zeros([mcs, steps+1], int)

  # Loop over Monte Carlo simulations
  for j in range(mcs):

    # Set initial position
    pos = start
    trajectories[j,0] = pos
    
    
    # Begin time evolution of the random walk
    i = 1
    while i < steps+1:

      # If particle is inside resetting region
      if pos <= reset_region:

        # Decide whether resetting occurs
        pos = resetting(pos, resetting_probability)
        trajectories[j,i] = pos

        # If resetting occurred (pos <= 0), trajectory ends
        if pos <= 0:
          break

        # Otherwise continue with hopping
        else: pass

      # If particle is outside resetting region, skip resetting
      else: pass

      # Perform hopping step
      pos = hopping(pos, case, bias_factor)
      trajectories[j,i] = pos

      # If recovery state is reached
      if pos <= 0:
          i += 1
          break

      # Otherwise continue the walk
      else: pass
      i+= 1

  # --------------------------------------------------------------------------
  # Save trajectories to CSV file
  # --------------------------------------------------------------------------

  Filepath = "" 
  + str(filename) + str(".csv")
  
  np.savetxt(Filepath, trajectories, delimiter = ",")

  return trajectories


# =============================================================================
# Recovery time calculation
# =============================================================================
def return_recovery_time(single_traj, reset_duration, h):

  """
  Computes recovery time from a single trajectory.

  Inputs:
   single_traj:      array containing particle positions through time
   reset_duration:   duration associated with a resetting event
   h:                time duration associated with each hop

  Returns:
   recovery_time:    time required for the particle to reach recovery
                     state (position 0)
  """

  recovery_time = 0

  # Iterate through trajectory positions
  for i in range(1, len(single_traj)):
    
    # Recovery via resetting
    if single_traj[i] == -1: 
      recovery_time = (i-1)/h + reset_duration
      break
      
    # Recovery via hopping
    elif single_traj[i] == 0:
      recovery_time = i/h
      break
      
    else: 
      pass

  return recovery_time


# =============================================================================
# Mean recovery time (MRT)
# =============================================================================
def return_mrt(case, bias_factor, resetting_probability, start, steps, 
               reset_region, mcs, reset_duration, h):

  """
  Computes the conditional Mean Recovery Time (MRT) using Monte Carlo
  simulations of the stochastic walk with resetting.

  Inputs:
   case:                  hopping rule
   bias_factor:           hopping bias strength
   resetting_probability: resetting probability inside reset region
   start:                 initial particle position
   steps:                 maximum simulation length
   reset_region:          region where resetting is allowed
   mcs:                   number of Monte Carlo simulations
   reset_duration:        time cost of resetting
   h:                     time duration of each hop

  Returns:
   conditional_mrt:       mean recovery time conditioned on successful
                          recovery events
  """

  # Array to store recovery time from each simulation
  recovery_time = np.zeros(mcs)

  # Loop through Monte Carlo runs
  for j in range(mcs):

    # Initialize position
    pos = start
    
    # Simulate trajectory
    i = 1
    while i < steps+1:

      # If inside resetting region
      if pos <= reset_region:

        # Attempt resetting
        pos = resetting(pos, resetting_probability)
        
        # If reset occurred
        if pos <= 0:
          recovery_time[j] = (i-1)/h + reset_duration
          break

        else: pass

      # If outside reset region
      else: pass

      # Perform hopping step
      pos = hopping(pos, case, bias_factor)

      # Check if recovery state reached
      if pos <= 0:
          i += 1
          recovery_time[j] = i/h
          break

      else: pass
      i+= 1

  # ----------------------------------------------------------------------
  # Compute conditional mean recovery time
  # (exclude simulations where recovery did not occur)
  # ----------------------------------------------------------------------

  conditional_mrt = np.mean(recovery_time[recovery_time !=0 ])

  return conditional_mrt
