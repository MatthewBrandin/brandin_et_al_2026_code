import numpy as np
from shapely.geometry import Polygon, box, LineString
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import scipy as sp
from scipy.optimize import minimize
import simplekml
from sklearn.mixture import GaussianMixture

def create_grid_rectangle(polygon, rect_width, rect_height):
    minx, miny, maxx, maxy = polygon.bounds
    grid_cells = []
    for x in np.arange(minx, maxx, rect_width):
        for y in np.arange(miny, maxy, rect_height):
            rectangle = box(x, y, x + rect_width, y + rect_height)
            grid_cells.append(rectangle)
    return grid_cells

def clip_rectangles_to_polygon(rectangles, polygon):
    clipped_rectangles = [rect.intersection(polygon) for rect in rectangles]
    return [rect for rect in clipped_rectangles if not rect.is_empty]

def keep_rectangles_touching_polygon(rectangles, polygon):
    touching_rectangles = [rect for rect in rectangles if not rect.intersection(polygon).is_empty]
    return touching_rectangles

def save_squares_new(clipped_squares):
    # This function saves the grid squares as .kml files for easy plotting

    output_dir = "grids"
    os.makedirs(output_dir, exist_ok=True)
    for i, square in enumerate(clipped_squares, start=1):
        kml = simplekml.Kml()

        # Extract coordinates from Shapely polygon
        coords = [(x, y) for x, y in square.exterior.coords]

        # Create a polygon
        pol = kml.newpolygon(name=f"grid_{i}", outerboundaryis=coords)

        # --- OGR-friendly styling ---
        pol.style.linestyle.color = simplekml.Color.red  # Red outline
        pol.style.linestyle.width = 2

        # Use mostly transparent fill (alpha 1 instead of 0) to avoid GDAL errors
        pol.style.polystyle.color = simplekml.Color.changealphaint(1, simplekml.Color.red)

        # Save
        kml_filename = os.path.join(output_dir, f"grid_{i}.kml")
        kml.save(kml_filename)
        print(f"Saved {kml_filename}")

def little_map(R1,R2,R3,R4, grid_num):
    # This function use GMT to create a tiny map of the grid square

    # get min and max topography
    os.system(f'gmt grd2xyz dem_cut.grd > temp.txt')
    topo = np.loadtxt('temp.txt')[:,2]
    t1 = min(topo)
    t2 = max(topo)
    dt = (t2-t1)/100

    os.system(f'gmt begin map_{grid_num} png')
    os.system(f'gmt makecpt -Cgray -T{t1}/{t2}/{dt} > topo.cpt')
    os.system(f'gmt grdimage dem.grd -Ctopo.cpt -R{R1}/{R2}/{R3}/{R4}')
    os.system(f'gmt basemap -Bxa0.1f0.05 -Bya0.1f0.05 -BWSne+t" "')
    os.system(f'gmt colorbar -Ctopo.cpt -Bx20f10+l" Grid {grid_num} Elev(m)"')
    os.system(f'gmt end')
    os.system(f'rm topo.cpt temp.txt') #clean up
    return

def get_acrit(R1,R2,R3,R4, grid_num):
    # this function calculates the critical acceleration of each point in the grid square

    #cut dem to size
    os.system(f'gmt grdcut dem.grd -R{R1}/{R2}/{R3}/{R4} -Gdem_cut.grd')
    #Optional: Get little map of the area
    #little_map(R1,R2,R3,R4,grid_num)
    #filter at 54 m
    os.system(f'gmt grdfilter dem_cut.grd -D1 -Fg0.054 -Ni -Gfilt.grd')
    #take gradient
    os.system(f'gmt grdgradient filt.grd -Sslope.grd -Da -Gdirection.grd')
    #convert to radians
    os.system(f'gmt grdmath slope.grd ATAN = slope.grd')
    #convert to xyz
    os.system(f'gmt grd2xyz slope.grd > slope.txt')
    os.system(f'gmt grd2xyz direction.grd > direction.txt')

    slope = np.loadtxt('slope.txt')
    direction = np.loadtxt('direction.txt')
    os.system(f'rm dem_cut.grd filt.grd slope.grd direction.grd direction.txt slope.txt') #clean up

    #organize
    lon = slope[:,0]
    lat = slope[:,1]
    alpha = slope[:,2]
    direction = direction[:,2]

    #constants
    c = 2 #cohesion of soil kPa
    gamma = 15.5 #unit weight of soil kN/m3
    t = 1 #soil thickness m
    f = 0.7 #friction coefficiant = tan(friction angle of repose)
    m = 0 #water saturation
    gamma_w = 9.81 #unit weight of water

    #calculate terms
    cohesion = c / (gamma * t * np.sin(alpha))
    friction = f/np.tan(alpha)
    water = (m*gamma_w*f)/(gamma*np.tan(alpha))

    #calculate static factor of saftey
    FS = cohesion + friction - water

    #if FS < 1, replace with 1.01
    FS[(FS < 1)] = 1.01

    #calculate critical acceleration
    ac = (FS - 1)*np.sin(alpha) #, in g's
    #ac = ac / K #apply topo amp

    # Propogate the Error? How? Important enough to spend time figuring out?

    #put all together into output file
    acrit = np.column_stack((lon, lat, ac, direction))

    np.savetxt('acrit.txt', acrit)
    print('ACRIT CALCULATED & SAVED')
    return

def get_data(R1,R2,R3,R4, grid_num):
    # This function calculates critical acceleration and InSAR change in Coherence for each point in grid square

    # GET ACRIT & CORR
    # use get_acrit to calculate acrit
    get_acrit(R1, R2, R3, R4, grid_num)  # calculates & saves acrit.txt as output
    # get data points from both before and after interferograms at those coords
    # before.grd and after.grd are InSAR corr_ll.grd files produced by GMTSAR
    os.system(f'gmt grdtrack acrit.txt -Gbefore.grd -Gafter.grd > data.txt')
    # clean up directory
    os.system(f'rm acrit.txt')
    # END GET ACRIT & CORR

    #check if there is no data
    #if no data, just return an empty array, later empty grid test will catch it and skip that grid
    if os.path.getsize('data.txt') == 0:
        return np.array([]), np.array([]), np.array([])

    # load data
    data = np.loadtxt(f'data.txt')
    # clean up directory
    os.system(f'rm data.txt')

    # get rid of nans, pixels with no data
    data = data[~np.isnan(data).any(axis=1), :]
    # organize
    lon = data[:, 0]
    lat = data[:, 1]
    acrit = data[:, 2]
    direction = data[:, 3]
    before = data[:, 4]
    after = data[:, 5]

    # calculate change in InSAR correlation
    corr = after - before #abs(after - before)

    return acrit, direction, corr

def grid_classifier(means):
    # This function classifies whether a PGA estimate is an upper bound, lower bound, or bounded value

    # need to sort
    # sort components by x-mean for consistent left/right labeling
    order = np.argsort(means[:, 0])
    i1, i2 = order[0], order[1]

    mean1_x, mean1_y = means[i1, 0], means[i1, 1]
    mean2_x, mean2_y = means[i2, 0], means[i2, 1]

    # if both means are lower than 0.5, then the terrain is too oversampled for steep slopes
    # this means we can only give a lower bound
    if (mean1_x < 0.5) and (mean2_x < 0.5):
        return 0
    # if both means are higher than 0.5, then the terrain is too oversampled for gentle slopes
    # this means we can only give an upper bound
    elif (mean1_x > 0.5) and (mean2_x > 0.5):
        return 1
    # if instead one mean is low and the other is high
    # and especially if the right mean is above the left mean, that means we have an even sample
    # this means we can give an approximate estimate for an exact value
    elif ((mean1_x < 0.5) and (mean2_x > 0.5)) and (mean2_y > mean1_y):
        return 2
    # otherwise, just assume its a lower bound
    else:
        return 0

def estimate_pga(acrit, corr, grid_num):
    # This Function uses a GMM to produce a bound on PGA given critical acceleration and Change in InSAR Coherence data

    # Organize data
    x = acrit
    y = corr
    data = np.column_stack((x, y))

    # Step 0: Apply Error as Noise in the data
    # ie, "smear" the data in the x direction
    # assume critical accelerations all have error of 0.15
    #N = 10 # number of Monte Carlo realizations
    #np.random.seed(42)
    #x_noisy = x + np.random.normal(0, 0.15)
    #data = np.column_stack((x_noisy, y))

    # Step 1: Fit a GMM, two 2-D Gaussians
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(data)

    # Step 1.5: classify this grid square
    grid_class = grid_classifier(gmm.means_)

    mu1, mu2 = gmm.means_
    Sigma1, Sigma2 = gmm.covariances_

    # Step 2: Assume equal covariances (LDA case)
    # Use the average covariance as a shared estimate
    Sigma = (Sigma1 + Sigma2) / 2

    # Compute weight vector and bias
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ (mu1 - mu2)
    b = -0.5 * (mu1.T @ Sigma_inv @ mu1 - mu2.T @ Sigma_inv @ mu2)

    # Step 3: Define decision boundary line
    # Equation: w_x * x + w_y * y + b = 0  →  y = (-b - w_x * x) / w_y
    x_vals = np.linspace(np.min(x) - 1, np.max(x) + 1, 200)
    y_vals = (-b - w[0] * x_vals) / w[1]

    # Step 4: Define the Y-Axis Margin
    y0 = (mu1[1] + mu2[1])/2
    #since y0 represents the threshold of Change in InSAR Coherence, it cannot be positive
    if y0 > 0:
        y0 = 0

    # Step 5: Find the intersection between decision boundary and marginal
    x_cross = -(w[1]*y0 + b)/w[0]

    # Step 6: (Optional) Plot result
    plt.figure(figsize=(8, 6))
    #plot data
    plt.scatter(x,y, alpha=0.4, color='gray')
    #plot GMM means
    plt.scatter(*mu1, color='red', marker='x', s=100)
    plt.scatter(*mu2, color='red', marker='x', s=100, label='Means')
    #plot decision boundary
    plt.plot(x_vals, y_vals, '--', label='Decision Boundary', color='blue')
    #plot marginal
    plt.axhline(y0, color='red', linestyle='--', label='Coherence Threshold')
    #plot intersection
    plt.axvline(x_cross, color='purple', linestyle=':', label='Bound')
    plt.xlabel('Critical Acceleration (g)')
    plt.ylabel('Change in InSAR Coherence')
    #plt.title(f'{grid_num}, {grid_class}')
    plt.xlim([0,0.9])
    plt.ylim([-1,1])
    plt.legend()
    #plt.savefig(f'fit_{grid_num}.png')
    #plt.show()

    #PGA obviously can't be negative, so zero is minimum
    #highest modeled PGA in this region was ~1.0g, which is maximum
    if x_cross < 0 or x_cross > 1.0:
        x_cross = np.nan

    return x_cross, grid_class

def model_value(R1,R2,R3,R4):
    # old depreciated function

    # use USGS model for now
    test_shift = 0.00
    os.system(f'gmt grd2xyz -R{R1+test_shift}/{R2+test_shift}/{R3}/{R4} PGA_scaled.grd > output.txt')

    # load data
    pga_actual = np.loadtxt(f'output.txt')

    # clean up directory
    os.system(f'rm output.txt')

    #if no data in this grid, return NaN
    if pga_actual.size == 0:
        pga_actual = np.nan
        return pga_actual

    #if only one data point in this grid, return that value
    if pga_actual.ndim == 1:
        pga_actual = pga_actual[2]
        return pga_actual

    #otherwise, take the mean of all points in this grid
    pga_actual = np.mean(pga_actual[:, 2])
    return pga_actual

def model_value_new(R1,R2,R3,R4, validation_model):
    # this function gets the average PGA reported by the Validation Model within a grid square

    lon = validation_model[:,0]
    lat = validation_model[:,1]
    vpga = validation_model[:,3] / 100 # data is in percent g

    # find the average pga and uncertainty within grid square
    mask = ((lon >= R1) & (lon <= R2) & (lat >= R3) & (lat <= R4))

    # If no data points fall inside the box, return np.nan
    if not np.any(mask):
        return np.nan

    vpga = np.mean(vpga[mask])

    return vpga

#validation model to compare against
validation_model = np.loadtxt('../EMC_combined_share/DATA.txt', delimiter=' ')

#read in kml boundary file as array of coordinates
os.system(f'gmt kml2gmt cucapah_1.kml > bound.txt')
coords_array = np.loadtxt(f'bound.txt', skiprows=1)
os.system(f'rm bound.txt')

#Convert the NumPy array to a list of tuples
coords = [tuple(coord) for coord in coords_array]

#convert to shapley polygon
polygon = Polygon(coords)
minx, miny, maxx, maxy = polygon.bounds
# Now you can proceed to create a grid and clip

grid_size_width = 0.018 # Adjust size as needed, 0.018 deg ~ 2 km
grid_size_height = grid_size_width
grid_squares = create_grid_rectangle(polygon, grid_size_width, grid_size_height)
clipped_squares = keep_rectangles_touching_polygon(grid_squares, polygon)

# uncomment to save squares as kmls
#save_squares_new(clipped_squares)

# Okay, now that we have grids, we can actually look at the data

# PGA ESTIMATION

#Cycle through each grid square, get data and estimate PGA for each one
grid_num = 1
values_lon = []
values_lat = []
values_est = []
values_actual = []
values_grid_num = []
values_typ = []

for square in clipped_squares:
    print(' ')
    print(f'Working on Grid {grid_num} of {len(clipped_squares)}')

    # Skip Selection
    #if grid_num != 118:
    #    print('SKIPPING')
    #    grid_num += 1
    #    continue

    # convert grid square into gmt coordinates
    lon, lat = square.exterior.xy
    R1 = min(lon)
    R2 = max(lon)
    R3 = min(lat)
    R4 = max(lat)
    values_grid_num.append(grid_num)     # record the number/name of the grid square
    values_lon.append(np.mean((R1, R2))) # record the center of the grid square
    values_lat.append(np.mean((R3, R4))) # record the center of the grid square
    print(R1, R2, R3, R4)

    # Do test grid
    #R1, R2, R3, R4 = -115.5753, -115.5504, 32.3987, 32.4170

    # get insar and topographic gradient data in that grid square
    acrit, direction, corr = get_data(R1, R2, R3, R4, grid_num)

    # if the grid is empty, or has almost no data, continue (gap in the data, estimate won't work)
    if len(acrit) < 10:
        print('EMPTY GRID')
        grid_num += 1
        values_est.append(np.nan)
        values_actual.append(model_value(R1,R2,R3,R4))
        values_typ.append(np.nan)
        continue

    # use data to estimate PGA
    pga_est, typ = estimate_pga(acrit, corr, grid_num)
    values_est.append(pga_est)
    values_typ.append(typ)

    # We need to compare our results to the validation model
    pga_actual = model_value_new(R1,R2,R3,R4, validation_model)
    values_actual.append(pga_actual)

    print(f'Estimated PGA: {pga_est} | Model PGA: {pga_actual} | Bound: {typ}')

    #PLOT
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))

    x1, y1 = polygon.exterior.xy
    x2, y2 = square.exterior.xy
    # axs[0].fill(x1, y1)
    for s in clipped_squares:
        x, y = s.exterior.xy
        axs[0].plot(x, y, color='gray')
    #for fault in all_faults:
    #    x, y = fault.xy
    #    axs[0].plot(x, y, color='red')
    axs[0].fill(x2, y2, color='orange')
    axs[0].grid()
    axs[0].set_title(f'Grid {grid_num}')
    axs[0].set_xlim(minx - 0.025, maxx + 0.025)
    axs[0].set_ylim(miny - 0.025, maxy + 0.025)

    xy = np.vstack([acrit, corr])
    z = sp.stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = acrit[idx], corr[idx], z[idx]
    scatter = axs[1].scatter(x, y, c=z, cmap='plasma')
    axs[1].set_xlim([0,0.9])
    axs[1].set_ylim([-1,1])
    axs[1].set_xlabel('Critical Acceleration (g)')
    axs[1].set_ylabel('Change in InSAR Coherence')
    fig.colorbar(scatter, ax=axs[1], label='Point Density')
    # plot the "real" PGA
    axs[1].axvline(x=pga_actual, color='green', linestyle='--', label='USGS PGA')
    # plot our estimate
    axs[1].axvline(x=pga_est, color='brown', linestyle='--', label='Estimate')
    axs[1].legend()

    axs[2].hist(acrit, bins=100, label='All Points')
    axs[2].set_xlabel('Critical Acceleration (g)')

    plt.tight_layout()
    #plt.savefig(f'grid_{grid_num}.png')
    plt.close('all')
    #plt.show()

    #incriment grid name
    grid_num += 1

plt.close('all')

# PLOT RESULTS

norm = Normalize(vmin=0.1, vmax=1.0)
cmap = cm.viridis

fig, ax = plt.subplots(2,2, figsize=(9, 9))
#plot the lower bounds
for square, value, type in zip(clipped_squares, values_est, values_typ):
    color = cmap(norm(value))
    x, y = square.exterior.xy
    ax[0,0].plot(x, y, color='gray')
    if type != 0:
        continue
    ax[0,0].fill(x,y,color=color)
ax[0,0].grid()
#ax[0,0].set_xlabel('longitude')
ax[0,0].set_ylabel('latitude')
ax[0,0].set_title('Lower Bound')
ax[0,0].set_xlim(minx-0.025, maxx+0.025)
ax[0,0].set_ylim(miny-0.025, maxy+0.025)

#plot the upper bounds
for square, value, type in zip(clipped_squares, values_est, values_typ):
    color = cmap(norm(value))
    x, y = square.exterior.xy
    ax[0,1].plot(x, y, color='gray')
    if type != 1:
        continue
    ax[0,1].fill(x,y,color=color)
ax[0,1].grid()
#ax[0,1].set_xlabel('longitude')
ax[0,1].set_ylabel('latitude')
ax[0,1].set_title('Upper Bound')
ax[0,1].set_xlim(minx-0.025, maxx+0.025)
ax[0,1].set_ylim(miny-0.025, maxy+0.025)

# plot the bounded values
for square, value, type in zip(clipped_squares, values_est, values_typ):
    color = cmap(norm(value))
    x, y = square.exterior.xy
    ax[1,0].plot(x, y, color='gray')
    if type != 2:
        continue
    ax[1,0].fill(x,y,color=color)
ax[1,0].grid()
ax[1,0].set_xlabel('longitude')
ax[1,0].set_ylabel('latitude')
ax[1,0].set_title('Bounded')
ax[1,0].set_xlim(minx-0.025, maxx+0.025)
ax[1,0].set_ylim(miny-0.025, maxy+0.025)

# plot the validation model values
for square, value, type in zip(clipped_squares, values_actual, values_typ):
    color = cmap(norm(value))
    x, y = square.exterior.xy
    ax[1,1].plot(x, y, color='gray')
    ax[1,1].fill(x,y,color=color)
ax[1,1].grid()
ax[1,1].set_xlabel('longitude')
ax[1,1].set_ylabel('latitude')
ax[1,1].set_title('Validation Model')
ax[1,1].set_xlim(minx-0.025, maxx+0.025)
ax[1,1].set_ylim(miny-0.025, maxy+0.025)

plt.subplots_adjust(wspace=0.25, hspace=0.25)

cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.7)
cbar.set_label('PGA (g)') # Add label to colorbar
cbar.ax.xaxis.set_ticks_position('bottom')  # Ensure ticks are at the bottom

plt.savefig(f'grid_all.png')


# RMS VALIDATION ANALYSIS
x = np.array(values_est)
y = np.array(values_actual)
t = np.array(values_typ)

# to do a proper fit, NaNs must be removed
x, y, t = x[~(np.isnan(x))], y[~(np.isnan(x))], t[~(np.isnan(x))]

x_exact, y_exact = x[np.where(t==2)], y[np.where(t==2)]
x_lower, y_lower = x[np.where(t==0)], y[np.where(t==0)]
x_upper, y_upper = x[np.where(t==1)], y[np.where(t==1)]

# only use the bounded values for RMS fit
rms = np.sqrt( sum( (y_exact-x_exact)**2 )/len(y_exact) )
print(f'RMS = {rms}')

plt.figure()
#plt.errorbar(x_exact, y_exact, yerr=s_exact, fmt='.', color='green')
#plt.errorbar(x_lower, y_lower, yerr=s_lower, fmt='.', color='blue')
#plt.errorbar(x_upper, y_upper, yerr=s_lower, fmt='.', color='red')
plt.scatter(x_exact, y_exact, color='green')
plt.scatter(x_upper, y_upper, color='red', alpha=0.5)
plt.scatter(x_lower, y_lower, color='blue', alpha=0.5)
plt.plot(x,x, color='orange') # slope intercept zero
plt.title(f'RMS = {np.round(rms, decimals=5)}')
plt.xlabel('PGA Bounds')
plt.ylabel('Validation Model')
plt.savefig(f'comp.png')
#plt.show()

# Save the Results
results = np.column_stack((values_lon, values_lat, values_grid_num, values_est, values_typ, values_actual))
np.savetxt('results.txt', results)