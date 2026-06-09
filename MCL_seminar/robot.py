import numpy as np
from shared import R

class Robot:
    __slots__ = ["I_xi"]
    
    def __init__(self, I_xi=np.zeros(3, dtype=float)):
        self.I_xi = I_xi
        
    def update_state(self, I_xi_dot, dt):
        self.I_xi += I_xi_dot * dt
        self.I_xi[2] = np.arctan2(np.sin(self.I_xi[2]), np.cos(self.I_xi[2]))
        
    def update_state_R(self, R_xi_dot, dt):
        I_xi_dot = R(self.I_xi[2]).T @ R_xi_dot
        self.update_state(I_xi_dot, dt)