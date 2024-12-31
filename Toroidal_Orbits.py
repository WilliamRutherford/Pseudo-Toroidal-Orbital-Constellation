import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.spatial import distance
from scipy.stats import linregress

'''
Given the length of the minor and major axes, generate the set of points for an ellipse centered on it's right focus.  
"divs" is the number of points created, unevenly spaced on the ellipse. 
'''
def generateEllipse(minor_len : float, major_len : float, divs : int = 100, log : bool = False, endpoint : bool = False):
    #if(minor_len > major_len):
        #print("minor length has exceeded major length")
    '''
    ecc = math.sqrt(1 - (minor_len / major_len)**2)
    ecc_sq = np.abs(1 - (minor_len / major_len)**2)
    '''
    # Get the angles 
    theta = np.linspace(0, 2*math.pi, num = divs, endpoint = endpoint)
    # x^2 / a^2 + y^2 / b^2 = 1
    # foci are at (c, 0) and (-c, 0) given c^2 = a^2 - b^2
    # (c,0) is right focus, (-c,0) is the left focus
    #pt_x = major_len * np.cos(theta) - np.sqrt(major_len**2 - minor_len**2)
    pt_x = major_len * np.cos(theta)
    pt_y = minor_len * np.sin(theta) 
    # Accounting for the possibility that the axes have flipped (minor len > major len)
    if(major_len > minor_len):
        pt_x -= np.sqrt(major_len**2 - minor_len**2)
    else:
        pt_y -= np.sqrt(minor_len**2 - major_len**2)

    if(log):
        print("x coord shape:", pt_x.shape)
        print("y coord shape:", pt_y.shape)
    return np.stack((pt_x, pt_y))

'''
Given a minor and major axis length for an ellipse, generate a set of points for an ellipse centered on it's right focus. 
The "Equal Area" part means that the area of each point and it's neighbouring point should be approximately equal.
This equal-area is done using least-squares optimization. 

minor_len: float64 
major_len: float64, greater than minor_len
divs: integer (number of points to generate) which should be an even number. 
return_angles : bool, tells us whether to also return the angles, not just the points

result_angles has shape [divs,]
result_pts: has shape [2, divs]

if(return_angles):
    return (result_angles, result_pts)
else:
    return result_pts
'''
def generateEqAreaEllipse(minor_len : float, major_len : float, divs : int = 100, log : bool = False, endpoint : bool = False, arc_calc_approx : bool = False, return_angles : bool = False):
    '''
    We start with the point closest to the origin, which is (major_len - c, 0) which is at angle = 0rad
    this starting point is not reflected. This leaves divs-1 points
    Split the ellipse into two mirrored halves (along the y-axis), each with (divs-1 // 2) points. 
    
    Each point will be uniquely represented by an angle t. for two consecutive points t_i and t_i++, what is the area of their triangle?
    p_i = ( major_len * np.cos(t_i) - np.sqrt(major_len**2 - minor_len**2)
            minor_len * np.sin(t_i))
    
    We can approximate equal area by taking an arc of angle (t_i++ - t_i) with radius ||p_i||
    this gives us an area of 1/2 * (t_i++ - t_i) * ||p_i||^2

    the length of one side is ||p_i||, the length of another side is ||p_i++||, and the interior angle is (t_i++ - t_i). this gives us a SAS triangle. 
    '''

    # A function that takes an angle theta, and gives us the point on the ellipse. 
    # t_i => p_i
    get_pt = parametricEllipse(minor_len, major_len)

    # t_i => ||p_i||^2
    def get_magn_sq(theta):
        return (major_len * np.cos(theta) - np.sqrt(major_len**2 - minor_len**2))**2 + (minor_len * np.sin(theta))**2

    start_pt = get_pt(0)
    end_pt   = get_pt(math.pi)

    ellipse_tot_area = minor_len * major_len * math.pi
    # area per piece = total area / divs
    area_per = ellipse_tot_area / divs

    def fit_fn(u):
        x = np.mod(u, 2*math.pi)
        # u has shape [divs,]
        # v has shape [divs,]
        v = np.roll(x, 1)
        # u_pts, v_pts has shape [2, divs]
        u_pts = get_pt(x)
        v_pts = get_pt(v)
        # det([[a,b],[c,d]]) = ad - bc
        # det([u, v]) = u_x * v_y - u_y * v_x
        all_areas = 1/2 * np.abs(u_pts[0] * v_pts[1] - u_pts[1] * v_pts[0])
        if(log):
            print("all areas shape:", all_areas.shape)
        #all_areas = np.array((divs))

        # These ones converge faster; I can only assume because it's a function in R^divs => R^divs, where changes are localized. 
        # variables x_i and x_i+1 affect x_i
        return np.abs(all_areas - area_per)
        #return np.abs(all_areas - np.mean(all_areas))

        # Both of the below are incredibly slow to converge, if at all.
        # This might be because they are a function of R^divs => 1 or 2, where we examine global changes and squish down.  
        # We want to minimize the absolute distance between the current area and the desired area, and the variance between each slice. 
        #return np.array([abs(ellipse_tot_area - np.sum(all_areas)), np.std(all_areas)])
        #return np.var(all_areas)
    
    result = least_squares(fit_fn, x0 = np.linspace(0, 2*math.pi, num = divs), bounds=(0, 2*math.pi))
    if(return_angles):
        return (result.x, get_pt(result.x))
    else:
        return get_pt(result.x)

def generateTorusCrossSection(outer_radius, divs = 99):
    circ_theta = np.linspace(0, 2*math.pi, divs)
    circ_x = outer_radius * np.cos(circ_theta) + 1
    if(outer_radius >= 1):
        circ_x = np.abs(circ_x)
    circ_y = outer_radius * np.sin(circ_theta)
    return np.vstack((circ_x, circ_y))    

### PARAMETRIC EQUATIONS ###

'''
Given the parameters of an ellipse, generate a parametric equation describing it. 

parametricEllipse(...) : t -> np.array((2,1))
or
parametricEllipse(...) : t = np.array((n,)) -> np.array((2,n))
t: either a single float, or a numpy array
'''
def parametricEllipse(minor_len, major_len, log=False):
    
    def create_ellipse(t):
        pt_x = major_len * np.cos(t)
        pt_y = minor_len * np.sin(t) 
        # Accounting for the possibility that the axes have flipped (minor len > major len)
        if(major_len > minor_len):
            pt_x -= np.sqrt(major_len**2 - minor_len**2)
        else:
            pt_y -= np.sqrt(minor_len**2 - major_len**2)

        if(log):
            print("x coord shape:", pt_x.shape)
            print("y coord shape:", pt_y.shape)
        return np.stack((pt_x, pt_y))
        
    return create_ellipse

'''
Given the parameters of an elliptical orbit, generate a parametric equation describing it. 

parametricOrbit(...) : t -> np.array((3,1))
or
parametricOrbit(...) : t = np.array((n,)) -> np.array((3,n))
t: either a single float, or a numpy array
'''
def parametricOrbit(minor_len, major_len, rot_ang, log=False):

    # We truncate the rotation matrix, since we assume the z component is 0. this will also project it into 3d for us. 
    rotation_matrix = R.from_euler('x', -rot_ang).as_matrix()[:,(0,1)]

    if(log):
        print("rotation matrix:", rotation_matrix)

    def create_orbit(t):
        return rotation_matrix @ np.vstack((major_len * math.cos(t), minor_len * math.sin(t)))

    return create_orbit

'''
Given the parameters of an ellipse, return a function that takes an angle and returns the polar radius.
This assumes the ellipse is centered on the right-most focus.
'''
def parametricEllipseRadius(minor_len, major_len, log=False):
    ecc = math.sqrt(1 - minor_len**2/major_len**2)
    
    def get_radius(theta): 
        return major_len * (1 - ecc**2) / (1 + ecc * np.cos(theta))
     
    return get_radius

'''
Given a singular ellipse and an inclination, generate an orbit.  
'''
def generateOrbit(ellipse, ang, log = False):
    rotation = R.from_euler('x', -ang)
    if(log):
        print("gen_orbit ellipse shape:", ellipse.shape)
        print("rotation matrix shape:", rotation.as_matrix().shape)
    return rotation.as_matrix() @ ellipse

'''
Given a set of 3d points, find their 2d cross-section using cylindrical coordinates. 
X: has shape [3, n]

Result R: has shape [2, n]
R[1] = X[2]
R[0,i] = sqrt(X[0,i,0]**2 + X[1,i,0])
'''
def crossSection(X):
    # the height of our cross-section is the z coordinate. 
    h_val = X[2]
    mag = np.linalg.norm(X[0:2], axis = 0)
    return np.vstack((mag, h_val))

'''
Given a 2D set of points, rotate it into the third (z) dimension by an angle (around the x axis), and find the cross-section. 
cross-section x = distance from the z' axis
cross-section y = distance projected onto z' axis
'''
def generateOrbitCrossSection(ellipse, angle, log = False):
    # Project our ellipse into 3D, keeping it in the x-y plane
    ellipse_3d = np.stack((ellipse[0], ellipse[1], np.zeros_like(ellipse[0])))
    # Form our unit vector z', which is in the y-z plane, the vector (0,0,1) rotated by our angle theta. 
    z_prime = np.array((0, np.sin(angle), np.cos(angle)))
    # Project each point on the ellipse onto z', to get the cross-section y (vertical) component. 
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

### 2D X-Y PLANE AREA COMPARISONS ###

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
Given a set of 2D points, calculate the area difference between them and a centered circle with radius "circle_radius". 
This treats the area difference as arcs; points further from the center are penalized more. 

X: a set of 2D points, centered at approximately "circle_center"
(circle_radius, circle_center): the parameters of the circle we compare to. 
absolute: whether or not we calculated the signed area (interior = negative, exterior = positive) or the absolute area (both are positive)  
vector_arcs: Whether to return the sum of all arc-areas, or the arc-areas themselves. 
'''
def hull_circle_diff(X, circle_radius, circle_center = np.array([1,0])[:, np.newaxis], log = False, absolute = True, vector_arcs = False):
    # Calculate the distance between each point, and the circle's center. 
    X_dist = np.linalg.norm(X - circle_center, axis = 0)
    # Subtract the circle radius, to get their difference
    # We take distances squared, to approximate the arc of a circle. (like pi * r^2)
    result = X_dist**2 - (circle_radius * np.ones_like(X_dist))**2
    if(absolute):
        result = np.abs(result)

    # What if we want to calculate the area as an arc?
    # The angular amount dedicated to each point, which normalizes the area for differing numbers of X points. 
    arc_ang = 2 * math.pi / X.shape[1]
    result = result / arc_ang

    if(log):
        print("X distance shape:", X_dist.shape)
        print("X distance sum:", np.sum(X_dist))
    if(vector_arcs):
        return result
    else:
        return np.sum(result)

'''
Given a set of 2D points, separate them into points inside and outside a circle with radius "circle_radius".
For each point, calculate ||x|| - circle radius.
Points outside contribute this to positive area, points inside contribute to negative area.

X: a set of 2D points, centered on the origin. 

return: (positive_area, negative_area) two floating point values, representing the differences in the exterior (positive) and interior (negative)
'''
def hull_signed_circle_dist(X, circle_radius, log = False):
    #pos_area = 0
    #neg_area = 0
    pt_magnitudes = np.linalg.norm(X, axis = 0)
    # We should split each of these into areas of an arc, since points further away from the center contain more area. 
    # Currently, it approximates theta * (k - j)
    # The area of an arc with angle theta for r in [j,k] is:
    # 1/2 * theta * (k^2 - j^2) = 1/2 * theta * (k + j) * (k - j)
    interior_pts = X[:, pt_magnitudes <= circle_radius]
    exterior_pts = X[:, pt_magnitudes >= circle_radius]
    num_interior = np.size(interior_pts, axis = 1)
    num_exterior = np.size(exterior_pts, axis = 1)    
    if(log):
        print("interior pts shape:", interior_pts.shape)
        print("exterior pts shape:", exterior_pts.shape)
    interior_tot = np.sum(np.linalg.norm(interior_pts, axis = 0))
    exterior_tot = np.sum(np.linalg.norm(exterior_pts, axis = 0))
    pos_area = exterior_tot  - circle_radius * num_exterior
    neg_area = -interior_tot + circle_radius * num_interior
    
    return (pos_area, neg_area)

def hull_signed_circle_area(X, circle_radius, log = False):
    # the length / magnitude for each point is:
    pt_magnitudes = np.linalg.norm(X, axis = 0)
    # the length / magnitude of the circle is always 'circle_radius'
    
### LEAST-SQUARES FITTING ###

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
        gen_cross_section = generateOrbitCrossSection(gen_ellipse, ang)
        # we should also add a term to make sure minor and major don't BOTH approach zero. 
        # we should also add a term to make sure minor_rad < major_rad. something like math.inf * (major_rad < minor_rad)
        return hull_diff(gen_cross_section, desired_cross)
    
    opt_result = least_squares(fit_func, [start_minor, start_major, start_ang])
    rad_a, rad_b, angle = opt_result.x
    min_rad = min(rad_a, rad_b)
    max_rad = max(rad_a, rad_b)
    return min_rad, max_rad, angle

'''
Given a desired circle with radius "circle_radius" around (1,0), find an ellipse orbit with a cross-section that approximates it (using least_squares)
'''
def fit_desired_circle(circle_radius, start_minor = 1, start_major = 1.0025, start_ang = math.pi/8, ellipse_divs = 100):
    # Given the parameters, find the cross-section and calculate the area to the desired
    def fit_func(u):
        #min_rad = min(u[0], u[1])
        #max_rad = max(u[0], u[1])
        min_rad = u[0]
        max_rad = u[1]
        ang     = u[2]
        gen_ellipse = generateEllipse(min_rad, max_rad, divs = ellipse_divs)
        gen_cross_section = generateOrbitCrossSection(gen_ellipse, ang)
        # we should also add a term to make sure minor and major don't BOTH approach zero. 
        # we should also add a term to make sure minor_rad < major_rad. something like math.inf * (major_rad < minor_rad)
        #return np.sum(hull_signed_circle_dist(gen_cross_section, circle_radius))
        return hull_circle_diff(gen_cross_section, circle_radius, vector_arcs=True)
    
    opt_result = least_squares(fit_func, [start_minor, start_major, start_ang])
    rad_a, rad_b, angle = opt_result.x
    #min_rad = min(rad_a, rad_b)
    #max_rad = max(rad_a, rad_b)
    min_rad = rad_a
    max_rad = rad_b
    return min_rad, max_rad, angle    

def fitTorusOrbit(out_radius, ellipse_divs = 100, cross_divs = 99):
    #desired_cross = generateTorusCrossSection(out_radius, cross_divs)
    
    # the starting parameters for fitting should match the Villerau circle. 
    
    # Find the difference between our generated ellipse cross-section and the desired cross-section
    #difference = hull_diff(cross_section, desired_cross, log = enable_logging)
    best_params = fit_desired_circle(out_radius, start_ang=np.arcsin(out_radius), ellipse_divs = ellipse_divs)
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

'''
Given a set of 3d points, rotate them around the z-axis to generate more copies
num_rots: the number of rotations to perform
matrix: if false, the resulting points are contiguous and have shape [3, n * num_rots]. 
        if true,  the resulting points have the shape [3, n, num_rots]
'''
def surfaceRevolution(in_pts, num_rots, log = False, matrix = True):
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



### PROPERTIES ###
    
enable_logging = False

# Parameters for the outputs / plots we will generate

test_ellipse = False
test_eq_area = False

single_orbit_plot = False
mult_orbit_plot = False

mult_orbit_density = False

relative_motion_plot = False
closer_points_plot = False

calc_hull_dev = False
calc_area = False
calc_signed_area = False

multi_radius_calc = True

# Some other calculations are dependent on having an ellipse of best fit. 
# this includes the variables best_ellipse, ellipse, and best_cross_section
fitting = not test_ellipse

circle_fitting = True

outer_radius = 0.3
num_objs = 25
ellipse_divs = 100
cross_divs = ellipse_divs

if(__name__ == "__main__"):
    if(test_ellipse):
        (min_rad, max_rad, ang) = (1, 3, math.pi/6)
        test_ellipse = generateEllipse(1, 2, log = enable_logging)
        print("test ellipse shape:", test_ellipse.shape)
        cross_section = generateOrbitCrossSection(test_ellipse, math.pi/6, log = enable_logging)
        ellipse = np.stack((test_ellipse[0], test_ellipse[1], np.zeros_like(test_ellipse[0])))
    
    # Generate the cross-section we want to fit to:
    desired_cross = generateTorusCrossSection(outer_radius, cross_divs)
    
    # Find the difference between our generated ellipse cross-section and the desired cross-section
    #difference = hull_diff(cross_section, desired_cross, log = enable_logging)
    if(fitting):
        if(circle_fitting):
            if(outer_radius >= 1):
                best_params = fit_desired_circle(outer_radius, start_ang=np.arcsin(1), ellipse_divs = ellipse_divs)
            else:
                best_params = fit_desired_circle(outer_radius, start_ang=np.arcsin(outer_radius), ellipse_divs = ellipse_divs)
        else:
            best_params = fit_desired(desired_cross, start_ang=np.arcsin(outer_radius), ellipse_divs = ellipse_divs)
        print("best parameters:", best_params)
        (min_rad, max_rad, ang) = best_params
        
        # We will also store these using more typical orbital mechanics terminology
        focal_dist = math.sqrt(max_rad**2 + min_rad**2)
        apoapsis  = max_rad + focal_dist
        periapsis = max_rad - focal_dist
        eccentricity = 1 - math.sqrt(min_rad**2 / max_rad**2)
        inclination = ang
        
        best_ellipse = generateEqAreaEllipse(min_rad, max_rad, divs = ellipse_divs)
        ellipse_angles = np.sort(np.arctan2(best_ellipse[1], best_ellipse[0]))
        best_cross_section = generateOrbitCrossSection(best_ellipse, ang)
        # turn the ellipse into 3d
        ellipse = np.stack((best_ellipse[0], best_ellipse[1], np.zeros_like(best_ellipse[0])))
        ellipse_flat = best_ellipse

    if(test_eq_area):
        major_len = max_rad
        minor_len = min_rad
        
        (pts_angles, pts) = generateEqAreaEllipse(minor_len, major_len, divs=ellipse_divs, return_angles = True)
        base_pts = generateEllipse(minor_len, major_len, divs = ellipse_divs, endpoint = True)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_aspect('equal')
        # Show the ellipse of equal area we generated
        ax1.scatter(pts[0], pts[1], zorder = 100)
        # Then, show a continuous line showing the outline of the ellipse
        ax1.plot(base_pts[0], base_pts[1], c = 'grey')
        # Also show the center point, for scale
        ax1.scatter(0,0, c='gray')
        # line for the x axis
        ax1.axhline(0, c = 'grey')
        #ax1.axline((0,0), slope = 0, c = 'grey')
        ax1.axvline(0, c = 'grey')
        #ax1.axline((0,0), slope = math.inf, c = 'grey')
        #counts, bins = np.histogram(np.arctan2(pts[1],pts[0]))
        #ax2.stairs(counts, bins)
        #pts_angles = np.arctan2(pts[1],pts[0])

        '''
        def get_pt(theta):
            pt_x = major_len * np.cos(theta) - np.sqrt(major_len**2 - minor_len**2)
            pt_y = minor_len * np.sin(theta)
            result = np.stack((pt_x, pt_y))
            # We need to make sure if given a single angle, we return an array of shape [2,1]. 
            # If we pass multiple angles, it will necessarily have the shape [2, n], with ndim == 2
            if(result.ndim == 1):
                result = result[:, np.newaxis]
            return result
        '''
        get_pt = parametricEllipse(minor_len, major_len)

        # Given a set of angles theta (representing points on an ellipse) calculate the area of the triangle including it's point and the next point. 
        def all_areas_gen(u):
            x = np.mod(u, 2*math.pi)
            # u has shape [divs,]
            # v has shape [divs,]
            v = np.roll(x, 1)
            # u_pts, v_pts has shape [2, divs]
            u_pts = get_pt(x)
            v_pts = get_pt(v)
            # det([[a,b],[c,d]]) = ad - bc
            # det([u, v]) = u_x * v_y - u_y * v_x
            all_areas = 1/2 * np.abs(u_pts[0] * v_pts[1] - u_pts[1] * v_pts[0])
            return all_areas
        
        all_areas = all_areas_gen(pts_angles)
        #counts, bins = np.histogram(all_areas)
        #ax2.stairs(counts, bins)
        ax2.hist(all_areas)
        # we would like to also show the desired area (ellipse_area / divs) on the histogram, but the scale on a histogram is way different. 
        #ax2.axline((major_len * minor_len * math.pi / (ellipse_divs), 0), slope = np.inf, c = 'grey')

        # Some useful outputs
        print("total area / actual area", np.sum(all_areas) / (math.pi * min_rad * max_rad))
        print("variance in area:", np.var(all_areas))

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
            curr_params = fit_desired_circle(curr_rad, start_ang=np.arcsin(curr_rad), ellipse_divs = ellipse_divs)
            (min_rad, max_rad, ang) = curr_params
            curr_ellipse = generateEllipse(min_rad, max_rad, divs = ellipse_divs)
            curr_cross_section = generateOrbitCrossSection(curr_ellipse, ang) 
            
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
    
    if(multi_radius_calc):
        test_num = 50
        # Generate all outer radii we will fit for:
        outer_radii = list(np.linspace(0.05, 0.90, test_num))

        # what values do we want to look at?
        # result from hull_circle_diff() or, the error of our minimization
        fit_error = []
        major_radii = []
        minor_radii = []
        angles = []

        for curr_rad in outer_radii:
            if(curr_rad >= 1):
                curr_params = fit_desired_circle(curr_rad, start_ang=np.arcsin(1), ellipse_divs = ellipse_divs)
            else:
                curr_params = fit_desired_circle(curr_rad, start_ang=np.arcsin(curr_rad), ellipse_divs = ellipse_divs)
            
            (min_rad, max_rad, ang) = curr_params
            # Calculate the error of our minimization, just like fit_desired_circle does
            gen_ellipse = generateEllipse(min_rad, max_rad, divs = ellipse_divs)
            gen_cross_section = generateOrbitCrossSection(gen_ellipse, ang)
            fit_error.append(hull_circle_diff(gen_cross_section, curr_rad))
            
            major_radii.append(max_rad)
            minor_radii.append(min_rad)
            angles.append(ang)

        major_radii = np.array(major_radii)

        #fig,(axr, axa, axe) = plt.subplots(1,3)
        fig, (axr, axa) = plt.subplots(1,2)

        axr.title.set_text("major & minor radius")
        axr.scatter(outer_radii, major_radii, c = 'blue')
        # The orbital period in terms of the central orbit (r = 1)
        # plt.scatter(outer_radii, major_radii**(3/2), c = 'red')
        axr.scatter(outer_radii, minor_radii, c = 'red')

        axa.title.set_text("Orbital Angle / Inclination")
        axa.scatter(outer_radii, angles)

        '''
        axe.title.set_text("Fit error hull_circle_diff")
        axe.scatter(outer_radii, fit_error)
        axe.axline(xy1 = (0,0), slope = 0, c = 'black')
        '''
        # We also want to fit a function to our "major radii" to see what pattern it follows. 
        major_fit, (resid, rank, sv, rcond) = np.polynomial.Polynomial.fit(outer_radii, major_radii, 3, full = True)
        major_coefs = major_fit.convert().coef
        print("outer-radius -> major_radius")
        print(major_fit.convert())

        fit_x, fit_y = major_fit.linspace(test_num, (np.min(outer_radii), np.max(outer_radii)))
        axr.plot(fit_x, fit_y, c = 'gray')
        # Calculate the R^2 value, which is 1 - SS_res / Variance
        print("R value:", 1 - resid[0] / np.var(major_radii))

        ang_fit, (ang_resid, ang_rank, ang_sv, ang_rcond) = np.polynomial.Polynomial.fit(outer_radii, angles, 2, full = True)
        print("outer-radius -> inclination")
        print(ang_fit.convert())
        ang_fix_x, ang_fit_y = ang_fit.linspace(test_num, (np.min(outer_radii), np.max(outer_radii)))
        axa.plot(ang_fix_x, ang_fit_y, c = 'gray')
        print("R value:", 1 - ang_resid[0] / np.var(angles))



    if(calc_area):
        tot_area = hull_circle_diff(best_cross_section, outer_radius, log = True)
        print("total area difference:", tot_area)
            
    if(calc_signed_area):
        cross_section_center = np.mean(best_cross_section, axis = 1)
        print("cross-section center:", cross_section_center)
        cross_centered = best_cross_section - cross_section_center[:, np.newaxis]
        (pos_area, neg_area) = hull_signed_circle_dist(cross_centered, outer_radius, log = True)
        print("positive (exterior) area:", pos_area)
        print("negative (interior) area:", neg_area)
    
    # Given our fit ellipse and an angle, plot some number of objs in offset orbits
    single_orbit = generateOrbit(ellipse, ang, log = enable_logging)
    
    #print("orbit shape:", single_orbit.shape)
    
    if(single_orbit_plot):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(single_orbit[0]), np.ptp(single_orbit[1]), np.ptp(single_orbit[2])))
        ax.scatter(single_orbit[0], single_orbit[1], single_orbit[2])
        ax.scatter(0,0,0, c='red')
        
    if(mult_orbit_density):

        all_orbits, _ = surfaceRevolution(single_orbit, num_objs, log = enable_logging, matrix=False)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.set_title('Multi-Orbit Point density (x,y)')
        h = ax1.hist2d(all_orbits[0], all_orbits[1], bins = 20)
        fig.colorbar(h[3], ax=ax1)
        #ax2.hist(ellipse_angles)

        get_pt = parametricEllipse(min_rad, max_rad)

        ax2.set_title("angle => angle increase (blue) \n angle => arc area vs desired (red)")
        ellipse_angles_diff = (ellipse_angles - np.roll(ellipse_angles,1))
        ax2.scatter(ellipse_angles[1:], ellipse_angles_diff[1:], c = 'blue')
        
        ax_area = ax2.twinx()
        u_pts = get_pt(ellipse_angles)
        v_pts = get_pt(np.roll(ellipse_angles, 1))
        # det([[a,b],[c,d]]) = ad - bc
        # det([u, v]) = u_x * v_y - u_y * v_x
        ellipse_area_diff = 1/2 * np.abs(u_pts[0] * v_pts[1] - u_pts[1] * v_pts[0])
        ellipse_area_var = ellipse_area_diff - min_rad * max_rad * math.pi / ellipse_divs
        ax_area.scatter(ellipse_angles, ellipse_area_var, c = 'red')



    if(mult_orbit_plot):
        all_orbits, labels = surfaceRevolution(single_orbit, num_objs, log = enable_logging, matrix=True)
        
        colors_theta = np.linspace(0, 2 * math.pi, num_objs)
        label_colors = np.stack((np.cos(colors_theta), np.sin(colors_theta), np.zeros_like(colors_theta))) / 2 + np.array([[1/2],[1/2],[0]])

        #label_colors = np.repeat(label_colors, num_objs, axis = 1)

        # now do a 3D plot for the orbits
        fig_mult = plt.figure('Multi-Orbit Plot', figsize = (14,6))
        ax1 = fig_mult.add_subplot(1, 2, 1, projection='3d')

        all_orbits_flat = np.reshape(all_orbits, (3, -1))
        
        ax1.set_box_aspect((np.ptp(all_orbits_flat[0]), np.ptp(all_orbits_flat[1]), np.ptp(all_orbits_flat[2])))
        if(True):
            ax1.scatter(all_orbits[0], all_orbits[1], all_orbits[2], c = labels)
        else:
            for i in range(0, num_objs):
                ax1.plot(all_orbits[0,:,i],all_orbits[1,:,i],all_orbits[2,:,i], c = label_colors[:,i])

        # do a 2D scatterplot for the y-z plane cross-section
        ax2 = fig_mult.add_subplot(1, 2, 2)
        ax2.set_aspect('equal')
        ax2.axline((1,0),(1.1,0), c='grey', zorder = 0)
        ax2.axline((1,0),(1,0.1), c='grey', zorder = 0)
        # Plot the desired cross first, so it's on the bottom.
        ax2.scatter(desired_cross[0], desired_cross[1], c = 'red')
        # Now, plot our best-fit cross-section over
        ax2.scatter(best_cross_section[0], best_cross_section[1], c = 'blue')
        
        # Plot the center of our best-fit cross-section
        ax2.scatter(np.mean(best_cross_section[0]), np.mean(best_cross_section[1]), c='blue', marker ='s')
        #plt.show()
    
    if(relative_motion_plot):
        num_tracked_objs = 5

        # Determines whether we want to consider all points on the circle, or just a slice enough to contain a full cross-section
        part_circle = False
        
        # the shape of all_orbits = [3, ellipse_divs, num_tracked_objs]
        all_orbits, labels = surfaceRevolution(single_orbit, num_tracked_objs, log = enable_logging, matrix = True)

        # Find equidistant points on the unit circle, which we will compare to the points
        if(not part_circle):
            theta = np.linspace(0, 2 * math.pi, ellipse_divs * num_tracked_objs, endpoint = False)
        else:
            theta = np.linspace(0, 2 * math.pi * 2 / num_tracked_objs, ellipse_divs * num_tracked_objs, endpoint = False)
       
        #theta = 2 * math.pi * theta_range / len(theta_range)
     
        circle_pts = np.stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        
        # For each point on the circle, find the closest point on each orbit.
        # There are 3 coordinates for each point, 'ellipse_divs' points on the circle, and 'num_tracked_objs' different orbits. 
        # Result has shape = [3, ellipse_divs, num_tracked_objs]
        circle_closest = np.zeros((3, len(theta), num_tracked_objs))
        
        # We also want to associate each resulting point with the theta value it is closest to. 
        #obj_labels = np.linspace(0, 1, num_tracked_objs)[np.newaxis, :]
        #angle_labels = np.repeat(obj_labels, repeats = len(theta), axis = 0)
        
        # angle_labels has shape (len(theta), num_tracked_objs)
        angle_labels = np.zeros((len(theta), num_tracked_objs))
        
        for i in range(0, num_tracked_objs):
            # curr orbit to check is number 'i'
            curr_orbit = all_orbits[:, :, i]
            # circle_dists has shape (len(theta), ellipse_divs)
            # circle_dists[i, j] is the distance between circle_pt[:, i] and curr_orbit[:, j]
            circle_dists = distance.cdist(circle_pts.T, curr_orbit.T)
            
            # For each point on the circle, find the closest point indx in the orbit. 
            # curr_closest_indx has shape: (len(theta),)
            # for each point on the circle, it's the index of the closest point on the current orbit. 
            curr_closest_indx = np.argmin(circle_dists, axis = 1)
            if(False and (i == 0)):
                print("curr_orbit shape is:", curr_orbit.shape)
                print("circle_pts shape is:", circle_pts.shape)
                print("circle_dists shape is:", circle_dists.shape)
                print("curr closest indx shape is:", curr_closest_indx.shape)
            curr_closest_pts = curr_orbit[:, curr_closest_indx]
            circle_closest[:, :, i] = curr_closest_pts
            #angle_labels[:, i] = theta[curr_closest_indx]
            angle_labels[:, i] = theta
        
        # angle_labels[i, :] = theta[i]
        
        # circle_closest has shape [3, len(theta), num_tracked_objs]
        
        # circle_closest[:, i, :] is the points in each orbit closest to circle_pts[:, i]
        # circle_closest axis 0: x or y coordinate
        # circle_closest axis 1: angle on circle it's closest to (theta <-> circle_pts[:, theta])
        # circle_closest axis 2: tracked object on orbit of         
        
        
        # circle_centers has shape [3, len(theta)]
        circle_centers = crossSection(np.mean(circle_closest, axis = 2))
        
        if(enable_logging):
            print("circle_closest shape:", circle_closest.shape)
            print("angle labels shape:", angle_labels.shape)
        fig_rel = plt.figure('Relative Orbital Motion', figsize = (14,6))
        ax_circ = fig_rel.add_subplot(1, 2, 1, projection='3d')
        
        circle_closest_flat = circle_closest.reshape((3, -1))
        angle_labels_flat   = angle_labels.reshape((1, -1))
        ax_circ.scatter(circle_pts[0], circle_pts[1], circle_pts[2], c = theta, cmap = 'viridis')
        ax_circ.set_box_aspect((np.ptp(circle_closest_flat[0]), np.ptp(circle_closest_flat[1]), np.ptp(circle_closest_flat[2])))
        ax_circ.scatter(circle_closest_flat[0], circle_closest_flat[1], circle_closest_flat[2], c = angle_labels_flat, cmap = 'viridis') 
        
        # Form a cross-section, of all points overlaid
        ax_cross = fig_rel.add_subplot(1, 2, 2)
        ax_cross.set_aspect('equal')
        ax_cross.axline((1,0),(1.1,0), c='grey', zorder = 0)
        ax_cross.axline((1,0),(1,0.1), c='grey', zorder = 0)            
        closest_cyl = crossSection(circle_closest_flat)
        ax_cross.scatter(closest_cyl[0], closest_cyl[1], c = angle_labels_flat, cmap = 'viridis')
        ax_cross.scatter(circle_centers[0], circle_centers[1], c = theta, cmap = 'viridis')
        
        # Plot the center of a single orbit for comparison
        single_orbit_center = np.mean(crossSection(single_orbit), axis = 1)
        ax_cross.scatter(single_orbit_center[0], single_orbit_center[1], c = 'red')   
        #plt.show()         
        
        # Also calculate the standard deviation for these avg points on the cross-section
        closest_cyl_deviation = np.std(closest_cyl, axis = 1) 
        print("Relative Motion cross-section std-deviation:", closest_cyl_deviation)
        
        # Also calculate the deviation laterally (angle before / after) of each corresponding sets of points 
        # Something like np.std(arctan(circle_closest, axis = 2))
        set_ang = np.arctan2(circle_closest[0], circle_closest[1])
        set_ang_avg_dev = np.mean(np.std(set_ang, axis = 1))
        print("Relative Motion mean angular std-deviation:", set_ang_avg_dev)
        
        # We can also calculate this with respect to circle_pts[:, i] (angle theta[i]) compared to points in circle_closest[:, :, i]
        
        # circle_pts[:, i] x [0,0,1] is the same as circle_pts[:, i] rotated by pi/2 (90 deg) in the z axis. (we will do it from x to y axis)
        # s = circle_pts @ [[0, -1],[1, 0]]
        # or s = np.stack((-circle_pts[1], circle_pts[0]))
        
        #perp_vec has shape = [2, len(theta)]
        perp_vec = np.stack((-circle_pts[1], circle_pts[0]))
        
        # relative_centered has shape = [2, len(theta), num_tracked_objs]
        relative_centered = circle_closest[0:2] - circle_pts[0:2, :, np.newaxis]

        fig, (axr, axf) = plt.subplots(1,2)
        axr.set_box_aspect(True)
        # This makes a cool rotationally symmetric pattern:
        axr.scatter(relative_centered[0], relative_centered[1], c = np.repeat(theta[:, np.newaxis], repeats = num_tracked_objs, axis = 1))   
        # the number of leaves is related to num_tracked_objs
        
        # Scalar Projection (dot product) of relative_centered onto perp_vec:
        # relative_centered[0, :, i] * perp_vec[0, :] + relative_centered[1, :, i] * perp_vec[1, :]
        # the result should have shape = [len(theta), num_tracked_objs]
        scalar_proj = relative_centered[0, :, :] * perp_vec[0, :, np.newaxis] + relative_centered[1, :, :] * perp_vec[1, :, np.newaxis]
        # proj_range has shape = (len(theta),)
        proj_range  = np.ptp(scalar_proj, axis = 1)
        proj_std    = np.std(scalar_proj, axis = 1)
        #has a similar fractal symmetry pattern to the one above
        axf.set_box_aspect(True)
        axf.scatter(theta, proj_range) 
        
        # maximum deviation:
        relative_max_diff = np.max(scalar_proj)
        
        if(closer_points_plot):
            # Right now, we are selecting a point from each separate orbit to compare. 
            # What if we look at all the points in the "closest" orbit, and see how many are closer than the closest points in other orbits?
            
            # We can just look at theta[0], or circle_pts[:, 0] which is [1, 0, 0]
            compare_circle = circle_pts[:, 0]
            # This is a point from each tracked orbit that is closest to circle_pts[:, 0] = [1, 0, 0]
            # compare_pts has shape = [3, num_tracked_objs]
            compare_pts = circle_closest[:, 0, :]
            # What is the second closest point? What is the furthest point?
            # Going with furthest:
            furthest_dist = np.max(distance.cdist(compare_pts.T, compare_circle[np.newaxis,:]))
            # Of all points, which is closer than the above?
            
            # For each tracked object, choose a colour. 
            tracked_color_ang = np.linspace(0, 2*math.pi, num_tracked_objs, endpoint = False)
            tracked_color = np.stack((np.cos(tracked_color_ang), np.sin(tracked_color_ang), np.zeros_like(tracked_color_ang))).T
            
            fig_closer = plt.figure('closer points')
            ax_cl = fig_closer.add_subplot(projection='3d')
            ax_cl.scatter(compare_circle[0], compare_circle[1], compare_circle[2], c = 'black')
            
            # Iterate through each tracked orbit, and plot those that are closer. 
            for i in range(0, num_tracked_objs):
                # calculate the distance to our one point
                curr_orbit = all_orbits[:, :, i]
                orbit_dist = np.linalg.norm(curr_orbit - compare_circle[:, np.newaxis], axis=0)
                curr_closer = curr_orbit[:, orbit_dist <= furthest_dist]
                ax_cl.scatter(curr_closer[0], curr_closer[1], curr_closer[2], tracked_color[i])
            
            
plt.show()