import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.spatial import distance




'''
Given the length of the minor and major axes, generate the set of points for an ellipse centered on it's right focus.  
"divs" is the number of points created, unevenly spaced on the ellipse. 
'''
def generateEllipse(minor_len, major_len, divs=100, log=False):
    #if(minor_len > major_len):
        #print("minor length has exceeded major length")
    '''
    ecc = math.sqrt(1 - (minor_len / major_len)**2)
    ecc_sq = np.abs(1 - (minor_len / major_len)**2)
    '''
    # Get the angles 
    theta = np.linspace(0, 2*math.pi, num = divs, endpoint = False)
    # x^2 / a^2 + y^2 / b^2 = 1
    # foci are at +/- c, which is +/- sqrt(a^2 - b^2)
    pt_x = major_len * np.cos(theta) + np.sqrt(major_len**2 - minor_len**2)
    pt_y = minor_len * np.sin(theta) 
    '''
    #In polar coordinates, the equation of the ellipse is:
    #pt_radius = minor_len / np.sqrt(1 - ecc_sq * np.cos(theta)**2)
    #pt_radius = minor_len / np.sqrt(1 - ecc_sq * np.cos(theta))
    pt_radius = minor_len * (1 - ecc_sq) / (1 - ecc * np.cos(theta))
    # We must also shift by c = sqrt(a^2 - b^2) so it is centered the left focus point. 
    #ellipse_c = np.sqrt(max(minor_len, major_len)**2 + min(minor_len, major_len)**2)
    ellipse_c = np.sqrt(major_len**2 - minor_len**2)
    #pt_x = pt_radius * np.cos(theta) + math.sqrt(major_len**2 - minor_len**2)
    pt_x = pt_radius * np.cos(theta)
    pt_y = pt_radius * np.sin(theta)
    '''
    if(log):
        print("x coord shape:", pt_x.shape)
        print("y coord shape:", pt_y.shape)
    return np.stack((pt_x, pt_y))

'''
Given a 2D set of points, rotate it into the third (z) dimension by an angle, and find the cross-section. 
cross-section x = distance from the z' axis
cross-section y = distance projected onto z' axis
'''
def generateCrossSection(ellipse, angle, log = False):
    # Project our ellipse into 3D, keeping it in the x-y plane
    ellipse_3d = np.stack((ellipse[0], ellipse[1], np.zeros_like(ellipse[0])))
    # Form our unit vector z', which is in the y-z plane, the vector (0,0,1) rotated by our angle theta. 
    z_prime = np.array((0, np.sin(angle), np.cos(angle)))
    # Project each point on the ellipse onto z', to get the cross-section y component. 
    yp_comp = z_prime @ ellipse_3d
    zp_proj = (z_prime[:, np.newaxis]) @ (yp_comp[:, np.newaxis]).T
    if(log):
        print("z_prime shape:", z_prime.shape)
        print("yp_comp / z-prime component shape:", yp_comp.shape)
        print("zp_proj shape:", zp_proj.shape)
    plane_comp = ellipse_3d - zp_proj
    xp_comp = np.linalg.norm(plane_comp, axis = 0)
    if(log):
        print("plane component shape:", plane_comp.shape)
        print("x-prime component shape:", xp_comp.shape)
    cross_section_pts = np.stack((xp_comp, yp_comp))
    return cross_section_pts

'''
Given two sets of 2D points, find an approximate difference between them.
For all points in a, find the distance to the closest point in b, and sum over all a values. 
'''
def hull_diff(a_pts, b_pts, log = False):
    dist_matr = distance.cdist(a_pts.T, b_pts.T)
    if(log):
        print("a_pts shape:", a_pts.shape, "b_pts shape:", b_pts.shape)
        print("distance matrix shape:", dist_matr.shape)
    # dist_matr[i, j] is the distance between a_pts[i] and b_pts[j]
    min_dist = np.min(dist_matr, axis=1)
    tot_dist = np.sum(min_dist)
    if(log):
        print("min distance from A shape:", min_dist.shape)
        print("total distance:", tot_dist)
    return tot_dist

'''
Given a desired cross-section, find an ellipse with a cross-section that approximates it (using least_squares)
'''
def fit_desired(desired_cross, start_minor = 0.45, start_major = 0.75, start_ang = math.pi / 8):
    # Given the parameters, find the cross-section and calculate the area to the desired
    def fit_func(u):
        #min_rad = min(u[0], u[1])
        #max_rad = max(u[0], u[1])
        min_rad = u[0]
        max_rad = u[1]
        ang     = u[2]
        gen_ellipse = generateEllipse(min_rad, max_rad, divs = 100)
        gen_cross_section = generateCrossSection(gen_ellipse, ang)
        # we should also add a term to make sure minor and major don't BOTH approach zero. 
        return hull_diff(gen_cross_section, desired_cross)
    
    opt_result = least_squares(fit_func, [start_minor, start_major, start_ang])
    rad_a, rad_b, angle = opt_result.x
    min_rad = min(rad_a, rad_b)
    max_rad = max(rad_a, rad_b)
    return min_rad, max_rad, angle
'''
Given a singular ellipse and an inclination, generate an orbit.  
'''
def generateOrbit(ellipse, ang, log = False):
    rotation = R.from_euler('x', -ang)
    if(log):
        print("gen_orbit ellipse shape:", ellipse.shape)
        print("rotation matrix shape:", rotation.as_matrix().shape)
    return rotation.as_matrix() @ ellipse

def surfaceRevolution(in_pts, num_rots, log = False, matrix = False):
    all_theta = np.linspace(0, 2*math.pi, num_rots, endpoint = False)
    all_rots = R.from_euler('z', all_theta)
    if(log):
        print("all rots shape:", all_rots.as_matrix().shape)
    all_orbits_first = all_rots.as_matrix() @ in_pts
    all_orbits = np.transpose(all_orbits_first, (1,2,0))
    if(log):
        print("all orbits shape:", all_orbits.shape)
    
    labels = np.arange(0, num_rots)

    
    # expand labels in axis = 1 according to the number of points in the ellipse
    ellipse_len = in_pts.shape[1]
    labels = np.resize(labels[np.newaxis, :], (ellipse_len, num_rots))
    print("labels shape:", labels.shape)
    
    
    if(matrix):
        return (all_orbits, labels)
    
    all_orbits_flat = np.reshape(all_orbits, (3,-1))
    labels_flat = labels.flatten('F')
    
    if(log):
        #print("(after reshape) all orbits shape:", all_orbits.shape)   
        print("all orbits flat shape:", all_orbits_flat.shape)
        print("labels flat shape:", labels_flat.shape)
    
    return (all_orbits_flat, labels_flat)
    

test_ellipse = False
fitting = True
single_orbit_plot = False

outer_radius = 0.7
num_objs = 25

if(__name__ == "__main__"):
    if(test_ellipse):
        (min_rad, max_rad, ang) = (1, 3, math.pi/6)
        test_ellipse = generateEllipse(1, 2, log=True)
        print("test ellipse shape:", test_ellipse.shape)
        cross_section = generateCrossSection(test_ellipse, math.pi/6, log = True)
        ellipse = np.stack((test_ellipse[0], test_ellipse[1], np.zeros_like(test_ellipse[0])))
    
    # Generate the cross-section we want to fit to:
    circ_theta = np.linspace(0, 2*math.pi, 99)
    circ_x = outer_radius * np.cos(circ_theta) + 1
    circ_y = outer_radius * np.sin(circ_theta)
    desired_cross = np.vstack((circ_x, circ_y))
    
    # the starting parameters for fitting should match the Villerau circle. 
    
    # Find the difference between our generated ellipse cross-section and the desired cross-section
    #difference = hull_diff(cross_section, desired_cross, log=True)
    if(fitting):
        best_params = fit_desired(desired_cross, start_minor = 1, start_major = 1 + outer_radius, start_ang=np.arcsin(outer_radius))
        print("best parameters:", best_params)
        (min_rad, max_rad, ang) = best_params
        best_ellipse = generateEllipse(best_params[0], best_params[1])
        best_cross_section = generateCrossSection(best_ellipse, best_params[2])
        # turn the ellipse into 3d
        ellipse = np.stack((best_ellipse[0], best_ellipse[1], np.zeros_like(best_ellipse[0])))
    
    # Given an ellipse and an angle, plot some number of objs in offset orbits
    single_orbit = generateOrbit(ellipse, ang, log = True)
    #print("orbit shape:", single_orbit.shape)
    if(single_orbit_plot):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(single_orbit[0]), np.ptp(single_orbit[1]), np.ptp(single_orbit[2])))
        ax.scatter(single_orbit[0], single_orbit[1], single_orbit[2])
        ax.scatter(0,0,0, c='red')
        
    else:
        all_orbits, labels = surfaceRevolution(single_orbit, num_objs, log = True, matrix=True)
        
        # now do a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(all_orbits[0]), np.ptp(all_orbits[1]), np.ptp(all_orbits[2])))
        ax.scatter(all_orbits[0], all_orbits[1], all_orbits[2], c = labels)
        
    