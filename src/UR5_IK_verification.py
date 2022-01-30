'''
# Inverse and Forward kinematic verification on UR5 manipulator

- Author: Hrushikesh Budhale (hbudhale@umd.edu)
'''
import sympy as sp
import numpy as np
from IPython.display import Math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import colors

'''
######### Tested on #########
# Python = 3.8.2            #
# matplotlib = 3.1.2        #
# sympy = 1.9               #
# numpy = 1.21.2            #
#############################
'''


# Defining Constants

pen_length = 0.0
d1 = 0.089159
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

a2 = -0.425
a3 = -0.39225

# Defining Symbols

th1, th2, th3, th4, th5, th6 = sp.symbols('\\theta_1^*, \\theta_2^*, \\theta_3^*, \\theta_4^*, \\theta_5^*, \\theta_6^*')
th = [th1, th2, th3, th4, th5, th6]     # List to store all theta symbols

T0, T1, T2, T3, T4, T5, T6= sp.symbols('T^0_0, T^0_1, T^0_2, T^0_3, T^0_4, T^0_5, T^0_6')
T0 = [T0, T1, T2, T3, T4, T5, T6]       # List to store all tranformation matrices from base link

Jv0, Jv1, Jv2, Jv3, Jv4, Jv5, Jv6 = sp.symbols('Jv0, Jv1, Jv2, Jv3, Jv4, Jv5, Jv6')
Jv = [Jv0, Jv1, Jv2, Jv3, Jv4, Jv5, Jv6]    # List to store all linear velocity jacobians for 6th link

Jw0, Jw1, Jw2, Jw3, Jw4, Jw5, Jw6 = sp.symbols('Jw0, Jw1, Jw2, Jw3, Jw4, Jw5, Jw6')
Jw = [Jw0, Jw1, Jw2, Jw3, Jw4, Jw5, Jw6]    # List to store all angular velocity jacobians for 6th link



# Defining necessary functions
def d_matrix(d):
    matrix = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
    return matrix

def theta_matrix(theta):
    matrix = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0, 0], [sp.sin(theta), sp.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return matrix

def r_matrix(r):
    matrix = sp.Matrix([[1, 0, 0, r], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return matrix

def alpha_matrix(alpha):
    matrix = sp.Matrix([[1, 0, 0, 0], [0, sp.cos(alpha), -sp.sin(alpha), 0], [0, sp.sin(alpha), sp.cos(alpha), 0], [0, 0, 0, 1]])
    return matrix

def get_transformation_matrix(d, theta, r, alpha):
    T = d_matrix(d)
    T = T @ theta_matrix(theta)
    T = T @ r_matrix(r)
    T = T @ alpha_matrix(alpha)
    return T

def get_dh_params(show_DH_table=False):
    Ts = sp.Matrix([[d1, th1,          0,  sp.pi/2],
                    [ 0, th2-sp.pi/2, a2,        0],
                    [ 0, th3        , a3,        0],
                    [d4, th4-sp.pi/2,  0,  sp.pi/2],
                    [d5, th5,          0, -sp.pi/2],
                    [d6, th6,          0,        0]])
    if show_DH_table:
        string = "$"
        string += " DH \\ parameters \\ for \\ UR5 \\ manipulator \\ are \\ as \\ follows, \\\ "
        string += "\\begin{aligned}"
        string += "\\begin{array}{cccc}"
        string += "d && \\theta && a && \\alpha \\\ "
        string += " \\hline \\\ "
        for i in range(int(len(Ts)/4)):
            for j in range(4):
                string += str(Ts[i,j]) + " && "
            string += " \\\ "
        string += "\\end{array}"
        string += "\\end{aligned}"
        string += "$"
        display(Math(string))
    
    return Ts

def get_fk(DH_params, print_intermediate_TF=False):
    '''
    Returns returns 
        1. transformation matrix for end effector pose w.r.t. origin
        2. list of transformation matrix for end effector w.r.t each 
    '''
    T0[0] = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    string = "$$ we \\ get \\ intermediate \\ transformation \\ matrices \\ as \\ follows, $$"
    for i in range(int(len(DH_params)/4)):
        Ti = get_transformation_matrix(d=DH_params[i, 0], theta=DH_params[i, 1], r=DH_params[i, 2], alpha=DH_params[i, 3])
        string += " \\ T_{}^{} = \\ ".format(i+1,i) + sp.latex(Ti)
        string += "\\\ "if i%3 == 2 else " \\ " # print on new line after printing 3 matrices
        T0[i+1] = T0[i] @ Ti
    string += "$"
    if print_intermediate_TF:
        display(Math(string))
    return T0

def get_Z(T_M):
    return T_M.col(2).row([0, 1, 2])

def get_O(T_M):
    return T_M.col(3).row([0, 1, 2])

def get_Jacobian_UR5(T0, method=1, simplifyMatrix=False):
    O6 = get_O(T0[6])
    for i in range(len(th)):
        Zi = get_Z(T0[i])
        # print(Zi)
        if method == 1:
            Oi = get_O(T0[i])
            ri = O6 - Oi
            Jv[i] = Zi.cross(ri)
        else:
            Jv[i] = sp.diff(O6, th[i])
        Jw[i] = Zi
    
    JV = sp.Matrix.hstack(Jv[0], Jv[1], Jv[2], Jv[3], Jv[4], Jv[5])
    JW = sp.Matrix.hstack(Jw[0], Jw[1], Jw[2], Jw[3], Jw[4], Jw[5])

    if simplifyMatrix:
        print("Simplifying Jacobian...")
        JV = sp.simplify(JV)
        JW = sp.simplify(JW)
        print("Complete.")
    
    J = sp.Matrix.vstack(JV, JW)
    return J
        
def get_T_for_circle(radius, theta):
    T0c1 = get_transformation_matrix(0.680, 0, 0, -sp.pi/2)
    T0c2 = get_transformation_matrix(0.605, theta, radius, 0)
    Tc = T0c1 @ T0c2
    return Tc

def draw_circle_using_jacobian(T0, J):
    theta_list = sp.Matrix([[-sp.pi/2], [-0.4+sp.pi/2], [-1.6], [-1.15+sp.pi/2], [-sp.pi/2], [-0.2]])
    dP = sp.Matrix([[0], [0], [0], [0], [0], [0]])
    diffP = [0]*6
    arc_angle = -sp.pi/2
    circle_radius = 0.1
    delta_t = 0.5

    Tc = get_T_for_circle(circle_radius, arc_angle)
    P_goal = Tc.row([0,1,2]).col([3])
    R_goal = Tc.row([0,1,2]).col([0,1,2])

    # Plotting related
    eff_pts = list()
    fig = plt.figure()
    ax = Axes3D(fig)

    while True:
        T_eff = T0[6].subs([(th1, theta_list[0]), (th2, theta_list[1]), (th3, theta_list[2]), 
                            (th4, theta_list[3]), (th5, theta_list[4]), (th6, theta_list[5])])
        P_eff = T_eff.row([0,1,2]).col([3])
        P_tip = P_eff + pen_length*(T_eff.row([0,1,2]).col([2]))
        R_curr = T_eff.row([0,1,2]).col([0,1,2])
        R60 = R_curr.transpose()
        R6e = R60 @ R_goal
        
        diffP[0] = P_goal[0] - P_tip[0]
        diffP[1] = P_goal[1] - P_tip[1]
        diffP[2] = P_goal[2] - P_tip[2]

        v = sp.acos((R6e[0,0]+R6e[1,1]+R6e[2,2]-1)/2)
        r = sp.Matrix([[0.],[0.],[0.]])
        r[0,0] = (R6e[2,1] - R6e[1,2])/(2*sp.sin(v))
        r[1,0] = (R6e[0,2] - R6e[2,0])/(2*sp.sin(v))
        r[2,0] = (R6e[1,0] - R6e[0,1])/(2*sp.sin(v))

        temp = r * sp.sin(v)
        diffP[3] = temp[0,0]
        diffP[4] = temp[1,0]
        diffP[5] = temp[2,0]

        
        dist = sp.sqrt(diffP[0]**2 + diffP[1]**2 + diffP[2]**2)
        if dist < 0.02:
            arc_angle += 0.1
            Tc = get_T_for_circle(circle_radius, arc_angle)
            P_goal = Tc.row([0,1,2]).col([3])
            R_goal = Tc.row([0,1,2]).col([0,1,2])
            if arc_angle > (3*sp.pi)/2 :
                break
        
        dP[0] = diffP[0]
        dP[1] = diffP[1]
        dP[2] = diffP[2]
        dP[3] = diffP[3] / 3.14
        dP[4] = diffP[4] / 3.14
        dP[5] = diffP[5] / 3.14
        
        Js = J.subs([(th1, theta_list[0]), (th2, theta_list[1]), (th3, theta_list[2]), 
                    (th4, theta_list[3]), (th5, theta_list[4]), (th6, theta_list[5])])
        J_inv = Js.inv('LU')
        dTh = (J_inv @ dP) * delta_t
        
        theta_list += np.clip(dTh, -0.1, 0.1)

        ax.clear()
        eff_pts = plot_figures(T0, theta_list, fig, ax, eff_pts)
        plt.pause(0.001)
        # break

    plt.show()
    return

def workspace_study_plot(T0):
    # Plotting related
    eff_pts = list()
    fig = plt.figure()
    ax = Axes3D(fig)
    theta_list = [0.0]*6
    for theta2 in np.linspace(0.1, 7*np.pi/8, 20):
        theta_list[1] = theta2
        for theta1 in np.linspace(0, 2*np.pi, 30):
            theta_list[0] = theta1
            ax.clear()
            eff_pts = plot_figures(T0, theta_list, fig, ax, eff_pts)
            plt.pause(0.001)
    plt.show()
    return
            
def plot_figures(T0, theta_list, fig, ax, eff_pts):
    # ax.axes.set_xlim3d(-1, 1) 
    # ax.axes.set_ylim3d(-1, 1) 
    # ax.axes.set_zlim3d(0, 1) 
    
    T = list()
    z0 = sp.Matrix([[0],[0],[1],[0]])
    
    for Ti in T0:
        T.append(Ti.subs([(th1, theta_list[0]), (th2, theta_list[1]), (th3, theta_list[2]), 
                          (th4, theta_list[3]), (th5, theta_list[4]), (th6, theta_list[5])]))

    # for i in range(len(T)-1):
    #     # Plot arm links
    #     ax.plot([T[i][0,3], T[i+1][0,3]],
    #             [T[i][1,3], T[i+1][1,3]],
    #             [T[i][2,3], T[i+1][2,3]], linewidth=3, color='b')

    #     # plot joint axes
    #     zi = (T[i] @ z0) * 0.05 
    #     ax.plot([T[i][0,3] + zi[0], T[i][0,3]], 
    #             [T[i][1,3] + zi[1], T[i][1,3]], 
    #             [T[i][2,3] + zi[2], T[i][2,3]], linewidth=5, color='r')

    # Draw Pen
    Te = T[-1]
    zi = (T[-1] @ z0) * pen_length
    eff_x = round(Te[0,3] + zi[0], 3)
    eff_y = round(Te[1,3] + zi[1], 3)
    eff_z = round(Te[2,3] + zi[2], 3)
    eff_pts.append([eff_x, eff_y, eff_z])
    ax.text(eff_x, eff_y, eff_z, '({}, {}, {})'.format(eff_x, eff_y, eff_z))
    ax.plot([Te[0,3], eff_x],
            [Te[1,3], eff_y],
            [Te[2,3], eff_z], linewidth=2, color='k')
    
    # Draw trajectory
    for eff_pt in eff_pts:
        ax.plot([eff_pt[0]],
                [eff_pt[1]],
                [eff_pt[2]], "o", color="g",  markersize=2)
    
    return eff_pts

# Calculate compute DH parameters for UR5
DH_params = get_dh_params()

# Get list of all transformations wrt origin (base frame)
T0 = get_fk(DH_params)

# Calculate Jacobian Matrix to use later
J = get_Jacobian_UR5(T0)

# draw_circle_using_jacobian(T0, J)
workspace_study_plot(T0)
