import numpy as np

### This file contains Arithmetic Carino Linking implementation for both return splits and active attrbution effects.


def carino_linking_return_split(r_s: np.array = None, r: np.array = None, R: float = 1):
    """
    Parameters:
    r_s (numpy.array): return componenent series 1-based
    r   (numpy.array): total_return series 1-based
    R   (float)      : total return compuonded over the period of the return series. 1-based

    Returns:
    float : the linked r_s

    """

    if R < 0:
        # invalid return: ignore the day and repalce with flat return (no market)
        R = 1

    # invalid return: ignore the day and repalce with flat return (no market)
    r = np.array([1 if r_i < 0 else r_i for r_i in r])
    r = np.array([1 if np.isnan(r_i) else r_i for r_i in r])

    r_s = np.array([1 if np.isnan(r_i) else r_i for r_i in r_s])

    k_t = np.zeros_like(r, dtype=float)  # Initialize k_t with the same shape as r
    for i in range(len(r)):
        if r[i] == 1:
            k_t[i] = 1
        else:
            k_t[i] = np.log(r[i]) / (r[i] - 1)

    if R == 1:
        K = 1
    else:
        K = np.log(R) / (R - 1)

    return sum(k_t * (r_s - 1)) / K + 1


def carino_linking_active(
    r_s: np.array = None,  # Active return component or attribution effect series
    r_p: np.array = None,  # Portfolio return series (1-based)
    r_b: np.array = None,  # Benchmark return series (1-based)
    Rp: float = 1,  # Total portfolio return compounded (1-based)
    Rb: float = 1,  # Total benchmark return compounded (1-based)
) -> float:
    """
    Carino linking algorithm using arithmetic linking for active returns (Rp - Rb) or attribution effects.

    Parameters:
    -----------
    r_s : np.array
        Active return component or attribution effect series (e.g., r_p - r_b or specific attribution effects).
        Not necessarily 1-based; typically arithmetic differences.
    r_p : np.array
        Portfolio return series (1-based).
    r_b : np.array
        Benchmark return series (1-based).
    Rp : float
        Total portfolio return compounded over the period (1-based).
    Rb : float
        Total benchmark return compounded over the period (1-based).

    Returns:
    --------
    float
        The linked active return component or attribution effect (arithmetic).
    """
    # Input validation and cleaning
    if Rp < 0 or Rb < 0:
        # Invalid total returns: replace with flat return (no market movement)
        Rp = 1
        Rb = 1

    # Clean the return series: replace negative or NaN values with flat return (1)
    r_p = np.array([1 if r_i < 0 or np.isnan(r_i) else r_i for r_i in r_p])
    r_b = np.array([1 if r_i < 0 or np.isnan(r_i) else r_i for r_i in r_b])

    # Clean r_s: replace NaN with 0 (neutral attribution effect)
    r_s = np.array([0 if np.isnan(r_i) else r_i for r_i in r_s])

    # Compute total arithmetic active return
    total_active_return = Rp - Rb

    k_t = np.zeros_like(r_p, dtype=float)  # Initialize k_t with the same shape as r
    for i in range(len(r_p)):
        if r_p[i] - r_b[i] == 0:
            k_t[i] = 1 / (r_p[i])
        else:
            k_t[i] = (np.log(r_p[i]) - np.log(r_b[i])) / (r_p[i] - r_b[i])

    if total_active_return == 0:
        K = 1
    else:
        K = (np.log(Rp) - np.log(Rb)) / total_active_return

    linked_effect = sum(k_t * r_s) / K
    return linked_effect


# # Example usage:
# if __name__ == "__main__":
#     # Example data
#     r_p = np.array([1.02, 1.03, 0.99])  # Portfolio returns (1-based)
#     r_b = np.array([1.01, 1.02, 1.00])  # Benchmark returns (1-based)
#     r_s = np.array([0.01, 0.01, -0.01])  # Active return components or attribution effects
#     Rp = np.prod(r_p)  # Total compounded portfolio return
#     Rb = np.prod(r_b)  # Total compounded benchmark return

#     result = carino_linking_active_return_split_arithmetic(r_s=r_s, r_p=r_p, r_b=r_b, Rp=Rp, Rb=Rb)
#     print(f"Linked active return component (arithmetic): {result}")
