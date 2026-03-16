import numpy as np
import random


# -------------------------------------------------------------------
# Basic hop function (biased random walk step)
# -------------------------------------------------------------------
def hop(pos, bias):
    """
    Performs one step of a 1D random walk with bias.

    Parameters
    ----------
    pos : int
        Current position of the walker.
    bias : float
        Bias towards the +1 direction.
        Positive bias -> more probability to move to the right
        Negative bias -> more probability to move to the left

    Returns
    -------
    int
        New position after the step.
    """

    # Weight (probability scale) for moving right (+1)
    w_1 = 50 + bias

    # Weight (probability scale) for moving left (-1)
    w_2 = 50 - bias

    # Randomly choose step direction according to weights
    # random.choices returns a list, hence k=1
    dpos = random.choices([1, -1], weights=[w_1, w_2], k=1)

    # Update position
    return pos + dpos[0]


# -------------------------------------------------------------------
# Hop with stochastic resetting
# -------------------------------------------------------------------
def hop_with_reset(pos, bias, reset_pos, reset_probability):
    """
    Performs one step of a biased random walk with a probability
    of resetting the walker to a fixed position.

    Parameters
    ----------
    pos : int
        Current position of the walker
    bias : float
        Bias toward the +1 direction
    reset_pos : int
        Position to which the walker resets
    reset_probability : float
        Relative weight of resetting compared to normal steps

    Returns
    -------
    int
        New position after the step (either +1, -1, or reset)
    """

    # Weight associated with resetting
    w_r = reset_probability

    # Adjust weights for left/right movement so total probability stays balanced
    # Reset probability reduces the probability of normal steps
    w_1 = 50 - 0.5 * w_r + bias
    w_2 = 50 - 0.5 * w_r - bias

    # If the walker is NOT already at the reset position,
    # allow three possibilities: +1 step, -1 step, or reset
    if pos != reset_pos:

        # Third option moves walker directly to reset_pos
        dpos = random.choices(
            [1, -1, -pos + reset_pos],  # displacement needed to reach reset_pos
            weights=[w_1, w_2, w_r],
            k=1
        )

    else:
        # If already at reset position, do not reset again
        # Only perform a normal biased step
        dpos = random.choices(
            [1, -1],
            weights=[50 + bias, 50 - bias],
            k=1
        )

    # Update position
    return pos + dpos[0]


# -------------------------------------------------------------------
# Monte Carlo trajectory generator
# -------------------------------------------------------------------
def get_trajectories(start_pos, bias, reset_pos, reset_probability,
                     max_resets, steps, mcs, filename):
    """
    Simulates multiple trajectories of a biased random walk with resetting.

    Parameters
    ----------
    start_pos : int
        Initial position of each walker
    bias : float
        Directional bias of the walk
    reset_pos : int
        Position where the walker resets
    reset_probability : float
        Weight controlling reset likelihood
    max_resets : int
        Maximum number of resets allowed per trajectory
    steps : int
        Number of time steps per trajectory
    mcs : int
        Number of Monte Carlo simulations (trajectories)
    filename : str
        Base filename used to save trajectory data

    Returns
    -------
    trajectories : ndarray
        Array of shape (mcs, steps+1) storing all trajectories
    resets : ndarray
        Number of resets that occurred in each trajectory
    """

    # Array storing positions for every trajectory
    # rows = trajectories, columns = time steps
    trajectories = np.zeros([mcs, steps + 1], int)

    # Array storing number of resets per trajectory
    resets = np.zeros(mcs, int)

    # Loop over Monte Carlo simulations
    for j in range(mcs):

        # Track number of steps taken
        step_count = 0

        # Track number of resets in this trajectory
        reset_count = 0

        # Set initial position
        trajectories[j, 0] = start_pos

        # Run trajectory evolution
        while step_count < steps:

            # Allow resetting only if maximum not reached
            if reset_count < max_resets:

                # Perform step with reset option
                trajectories[j, step_count + 1] = hop_with_reset(
                    trajectories[j, step_count],
                    bias,
                    reset_pos,
                    reset_probability
                )

                # Check if reset occurred
                if trajectories[j, step_count + 1] == reset_pos:
                    reset_count += 1

                step_count += 1

            else:
                # If maximum resets reached, continue normal walk
                trajectories[j, step_count + 1] = hop(
                    trajectories[j, step_count],
                    bias
                )

                step_count += 1

        # Store total resets for this trajectory
        resets[j] = reset_count


    # -------------------------------------------------------------------
    # Save trajectories to file
    # -------------------------------------------------------------------

    # Construct filename with bias information
    Filepath = filename + str("_bias_") + str(bias) + str(".csv")

    # Save trajectory matrix to CSV
    np.savetxt(Filepath, trajectories, delimiter=",")

    return trajectories, resets


# -------------------------------------------------------------------
# Example simulation
# -------------------------------------------------------------------

# Generate 10 trajectories of 100 steps each
trial = get_trajectories(
    5,      # start_pos
    0,      # bias
    5,      # reset_pos
    5,      # reset_probability
    100,    # max_resets
    100,    # steps
    10,     # mcs
    "trial" # filename
)

# Average number of resets across trajectories
np.mean(trial[1])


# -------------------------------------------------------------------
# Example of single step with reset
# -------------------------------------------------------------------

# pos, bias, reset_pos, reset_probability
hop_with_reset(10, -10, 5, 90)
