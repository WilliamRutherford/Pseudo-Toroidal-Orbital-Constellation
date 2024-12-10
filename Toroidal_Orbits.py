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

def generateTorusCrossSection(outer_radius, divs = 99):
    circ_theta = np.linspace(0, 2*math.pi, divs)
    circ_x = outer_radius * np.cos(circ_theta) + 1
    circ_y = outer_radius * np.sin(circ_theta)
    return np.vstack((circ_x, circ_y))    

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
    '''
    Could this be done in a single operation?
    '''
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
Given a set of 2D points, separate them into points inside and outside a circle with radius "circle_radius".
For each point, calculate ||x|| - circle radius.
Points outside contribute this into the positive, points inside contribute this to the negative. 

X: a set of 2D points, centered on the origin. 

return: (positive_area, negative_area) two floating point values, representing the differences in the exterior (positive) and interior (negative)
'''
def hull_signed_circle_dist(X, circle_radius, log = False):
    #pos_area = 0
    #neg_area = 0
    pt_magnitudes = np.linalg.norm(X, axis = 0)
    interior_pts = X[:, pt_magnitudes <= circle_radius]
    exterior_pts = X[:, pt_magnitudes >= circle_radius]
    num_interior = np.size(interior_pts, axis = 1)
    num_exterior = np.size(exterior_pts, axis = 1)    
    if(log):
        print("interior pts shape:", interior_pts.shape)
        print("exterior pts shape:", exterior_pts.shape)
    interior_tot = np.sum(np.linalg.norm(interior_pts, axis = 0))
    exterior_tot = np.sum(np.linalg.norm(exterior_pts, axis = 0))
    pos_area = exterior_tot - circle_radius * num_exterior
    neg_area = interior_tot - circle_radius * num_interior
    
    return (pos_area, neg_area)

'''
Given a desired cross-section, find an ellipse with a cross-section that approximates it (using least_squares)
'''
def fit_desired(desired_cross, start_minor = 1, start_major = 1.0025, start_ang = math.pi / 8, ellipse_divs = 100):
    # Given the parameters, find the cross-section and calculate the area to the desired
    def fit_func(u):
        #min_rad = min(u[0], u[1])
        #max_rad = max(u[0], u[1])
        min_rad = u[0]
        max_rad = u[1]
        ang     = u[2]
        gen_ellipse = generateEllipse(min_rad, max_rad, divs = ellipse_divs)
        gen_cross_section = generateCrossSection(gen_ellipse, ang)
        # we should also add a term to make sure minor and major don't BOTH approach zero. 
        # we should also add a term to make sure minor_rad < major_rad. something like math.inf * (major_rad < minor_rad)
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

def fitTorusOrbit(out_radius, ellipse_divs = 100, cross_divs = 99):
    desired_cross = generateTorusCrossSection(out_radius, cross_divs)
    
    # the starting parameters for fitting should match the Villerau circle. 
    
    # Find the difference between our generated ellipse cross-section and the desired cross-section
    #difference = hull_diff(cross_section, desired_cross, log = enable_logging)
    best_params = fit_desired(desired_cross, start_ang=np.arcsin(out_radius), ellipse_divs = ellipse_divs)
    #print("best parameters:", best_params)
    (min_rad, max_rad, ang) = best_params
    
    # We will also store these using more typical orbital mechanics terminology
    focal_dist = math.sqrt(max_rad**2 + min_rad**2)
    apoapsis  = max_rad + focal_dist
    periapsis = max_rad - focal_dist
    eccentricity = 1 - math.sqrt(min_rad**2 / max_rad**2)
    inclination = ang
    
    result = {
        "minor radius": min_rad,
        "major radius": max_rad,
        "inclination":  ang,
        "apoapsis":     apoapsis,
        "periapsis":    periapsis,
        "eccentricity": eccentricity
        }
    return result

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
    if(log):
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
    
enable_logging = False


# Parameters for the outputs / plots we will generate

test_ellipse = False

single_orbit_plot = False
mult_orbit_plot = True

calc_hull_dev = False
calc_signed_area = True

# Some other calculations are dependent on having an ellipse of best fit.
# this includes the variables best_ellipse, ellipse, and best_cross_section
fitting = False or (single_orbit_plot or mult_orbit_plot) or calc_signed_area



outer_radius = 0.9
num_objs = 25
cross_divs = 99

if(__name__ == "__main__"):
    if(test_ellipse):
        (min_rad, max_rad, ang) = (1, 3, math.pi/6)
        test_ellipse = generateEllipse(1, 2, log = enable_logging)
        print("test ellipse shape:", test_ellipse.shape)
        cross_section = generateCrossSection(test_ellipse, math.pi/6, log = enable_logging)
        ellipse = np.stack((test_ellipse[0], test_ellipse[1], np.zeros_like(test_ellipse[0])))
    
    # Generate the cross-section we want to fit to:
    desired_cross = generateTorusCrossSection(outer_radius, cross_divs)
    
    # the starting parameters for fitting should match the Villerau circle. 
    
    # Find the difference between our generated ellipse cross-section and the desired cross-section
    #difference = hull_diff(cross_section, desired_cross, log = enable_logging)
    if(fitting):
        best_params = fit_desired(desired_cross, start_ang=np.arcsin(outer_radius))
        print("best parameters:", best_params)
        (min_rad, max_rad, ang) = best_params
        
        # We will also store these using more typical orbital mechanics terminology
        focal_dist = math.sqrt(max_rad**2 + min_rad**2)
        apoapsis  = max_rad + focal_dist
        periapsis = max_rad - focal_dist
        eccentricity = 1 - math.sqrt(min_rad**2 / max_rad**2)
        inclination = ang
        
        best_ellipse = generateEllipse(min_rad, max_rad)
        best_cross_section = generateCrossSection(best_ellipse, ang)
        # turn the ellipse into 3d
        ellipse = np.stack((best_ellipse[0], best_ellipse[1], np.zeros_like(best_ellipse[0])))
        ellipse_flat = best_ellipse
    
    if(calc_hull_dev):
        test_num = 20
        # Generate all outer radii we will fit for:
        outer_radii = list(np.linspace(0.05, 0.95, test_num))
        
        # What results do we want to return?
        each_center = []
        each_center_diff = []
        each_hull_dev = []
        # What is the process we follow for each radius?
        
        for curr_rad in outer_radii:
            curr_desired_cross = generateTorusCrossSection(curr_rad)
            curr_params = fit_desired(curr_desired_cross, start_ang=np.arcsin(curr_rad))
            (min_rad, max_rad, ang) = curr_params
            curr_ellipse = generateEllipse(min_rad, max_rad)
            curr_cross_section = generateCrossSection(curr_ellipse, ang) 
            
            if(False):
                plt.scatter(curr_cross_section[0], curr_cross_section[1], c = np.array([curr_rad]*100))
            
            # Get the difference between the average of the current cross-section and the center at (1,0)
            curr_avg_center = np.mean(curr_cross_section, axis = 1)
            if(enable_logging):
                print("curr avg center:", curr_avg_center)
            each_center.append(curr_avg_center)
            # Also calculate the magnitude of their difference, to show how far it deviates. This is proportional to the outer radius. 
            each_center_diff.append(np.linalg.norm(curr_avg_center - np.array([1,0])))
            
            # Calculate the total distance between each point in our cross-section, and the closest point on our desired cross-section (hull difference)
            curr_hull_dev = hull_diff(curr_cross_section, curr_desired_cross)
            each_hull_dev.append(curr_hull_dev)
        plt.scatter(outer_radii, each_hull_dev)
            
    if(calc_signed_area):
        cross_section_center = np.mean(best_cross_section, axis = 1)
        print("cross-section center:", cross_section_center)
        cross_centered = best_cross_section - cross_section_center[:, np.newaxis]
        (pos_area, neg_area) = hull_signed_circle_dist(cross_centered, outer_radius, log = True)
        print("positive (exterior) area:", pos_area)
        print("negative (interior) area:", neg_area)
    
    # Given an ellipse and an angle, plot some number of objs in offset orbits
    if(single_orbit_plot or mult_orbit_plot):
        single_orbit = generateOrbit(ellipse, ang, log = enable_logging)
    #print("orbit shape:", single_orbit.shape)
    
    if(single_orbit_plot):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(single_orbit[0]), np.ptp(single_orbit[1]), np.ptp(single_orbit[2])))
        ax.scatter(single_orbit[0], single_orbit[1], single_orbit[2])
        ax.scatter(0,0,0, c='red')
        
    if(mult_orbit_plot):
        all_orbits, labels = surfaceRevolution(single_orbit, num_objs, log = enable_logging, matrix=True)
        
        # now do a 3D plot
        fig = plt.figure(figsize = (14,6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        ax1.set_box_aspect((np.ptp(all_orbits[0]), np.ptp(all_orbits[1]), np.ptp(all_orbits[2])))
        ax1.scatter(all_orbits[0], all_orbits[1], all_orbits[2], c = labels)
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect(1)
        ax2.axline((1,0),(1.1,0), c='grey', zorder = 0)
        ax2.axline((1,0),(1,0.1), c='grey', zorder = 0)
        # Plot the desired cross first, so it's on the bottom.
        ax2.scatter(desired_cross[0], desired_cross[1], c = 'red')
        # Now, plot our best-fit cross-section over
        ax2.scatter(best_cross_section[0], best_cross_section[1], c = 'blue')
        
        # Plot the center of our best-fit cross-section
        ax2.scatter(np.mean(best_cross_section[0]), np.mean(best_cross_section[1]), c='blue', marker ='s')
        
        
        
        
    