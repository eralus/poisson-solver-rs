import numpy as np
import multiprocessing as mp
import glob
import pyvista as pv
from ntcad import wannier90

def _get_clip_coords(start: np.ndarray, end: np.ndarray, tot_dims: np.ndarray) -> np.ndarray:
    assert((start<end).all())
    assert((end-start <= tot_dims).all())
    
    xrange = np.array(range(start[0], end[0]),dtype=np.int32)
    yrange = tot_dims[0]*np.array(range(start[1],end[1]),dtype=np.int32).reshape((-1,1))
    xy_pts = (xrange+yrange).reshape(-1)
    
    zrange = tot_dims[0]*tot_dims[1]*np.array(range(start[2],end[2]),dtype=np.int32).reshape((-1,1))

    return (zrange + xy_pts).reshape(-1)


# generate_mesh is meant to output a pyvista tetrahedral mesh that can be used for the poisson solver
def generate_mesh(
    path_to_xsf: str,
    propagation_axis:np.int_=0,
    cells_per_xsf:np.int_=5,
    n_cells_in_domain:np.int_=12,
    wf_shifts:np.ndarray=None,
    clip_start:np.ndarray=np.array([0,0,0]),
    clip_end:np.ndarray=None,
    samples_per_cell:np.int_=2000,
    sampling_prob_baseline:np.double=0.000,
    sampling_prob_sq_val:np.double=0.0,
    sampling_prob_sq_diff:np.double=0.0,
    sampling_prob_2nd_diff:np.double=1.0,
    exponent_2nd_diff:np.double=0.3
):
    print("Loading Wannier functions from ", path_to_xsf, ":", end='')
    if(path_to_xsf[-1] != '/'): path_to_xsf += '/'
    with mp.Pool() as pool:
        wfs = pool.map(
            wannier90.io.read_xsf,
            glob.glob(path_to_xsf + "*.xsf")
        )
    n_wfs = len(wfs)
    print("done!")
    
    print("Generating clipped Uniform grid: ", end='')
    shape = wfs[0].attr["3D_field"]["shape"]
    grid = pv.ImageData(
        dimensions=shape,
        spacing=[1./(s-1) for s in shape]
    )
    wannier_cell = wfs[0].attr["3D_field"]["cell"]
    origin = wfs[0].attr["3D_field"]["origin"]
    
    transform = np.eye(4)
    transform[:3, :3] = wannier_cell.T
    transform[:3, 3]  = origin

    grid = grid.transform(transform, inplace = False)
    if(clip_end is None): clip_end = np.array(shape)
    clipping_mask = _get_clip_coords(clip_start, clip_end, np.array(shape))
    print("done!")

    print("Getting_wf_data, shifting wfs: ", end='')
    wfs_data = np.zeros((n_wfs, clipping_mask.shape[0]))
    
    if wf_shifts is None:    
        wf_shifts = np.zeros(n_wfs)
    wf_shifts = wf_shifts.astype(np.int32)
    assert(min(wf_shifts) >= -1 and max(wf_shifts) <= 1)
    assert(wf_shifts.shape[0] == n_wfs)
    shift_pt_dist = int(shape[propagation_axis]/cells_per_xsf)
    coord_shift = np.zeros(3, dtype=np.int32)
    coord_shift[propagation_axis] = shift_pt_dist
    if max(abs(wf_shifts)) > 0:
        fwd_coords = _get_clip_coords(coord_shift,clip_end-clip_start,clip_end-clip_start)
        bwd_coords = _get_clip_coords(np.zeros(3,dtype=np.int32),clip_end-clip_start-coord_shift,clip_end-clip_start)
    for wf_idx in range(n_wfs):
        if wf_shifts[wf_idx] == -1:
            wfs_data[wf_idx,bwd_coords] = wfs[wf_idx].attr["3D_field"]["data"][clipping_mask][fwd_coords]
        elif wf_shifts[wf_idx] == 1:
            wfs_data[wf_idx,fwd_coords] = wfs[wf_idx].attr["3D_field"]["data"][clipping_mask][bwd_coords]
        else:
            wfs_data[wf_idx,:] = wfs[wf_idx].attr["3D_field"]["data"][clipping_mask]
    print("done!")
    
    print("Caluclating differentials: ", end='')
    n_pts_per_cell = int(clipping_mask.shape[0]/cells_per_xsf)
    
    wfs_data_reorganized = np.zeros_like(wfs_data)
    for i in range(cells_per_xsf):
        idxs = _get_clip_coords((np.zeros(3)+i*coord_shift).astype(np.int32),(clip_end-clip_start-(cells_per_xsf-1-i)*coord_shift).astype(np.int32),clip_end-clip_start)
        wfs_data_reorganized[:,i*n_pts_per_cell:(i+1)*n_pts_per_cell] = wfs_data[:,idxs]

    assert((clip_end-clip_start)[propagation_axis] % cells_per_xsf == 0)
    cell_shape = clip_end-clip_start
    cell_shape[propagation_axis] /= cells_per_xsf
    diff_start = np.zeros(3, dtype=np.int32)
    diff_end = cell_shape - (np.arange(3) != propagation_axis)
    diff_coords = np.zeros((5, np.prod(diff_end)), dtype=np.int32)
    diff_coords[0,:] = _get_clip_coords(diff_start, diff_end, cell_shape)
    diff_coords[4,:] = _get_clip_coords(diff_start+(np.arange(3)!=propagation_axis),cell_shape,cell_shape)
    for axis in range(1,4):
        if axis-1 == propagation_axis:
            shift = (np.arange(3) == (axis-1))
            relative_clip_start = _get_clip_coords(diff_start, diff_end-shift, diff_end-diff_start)
            diff_coords[axis,relative_clip_start] = _get_clip_coords(diff_start+shift, diff_end, cell_shape)
            relative_clip_end = _get_clip_coords((diff_end[axis-1]-1)*shift, diff_end, diff_end-diff_start)
            diff_coords[axis, relative_clip_end] = _get_clip_coords(diff_start, shift + diff_end*(np.arange(3) != (axis-1)), cell_shape) + n_pts_per_cell

        else:
            shift = (np.arange(3) == (axis-1))
            diff_coords[axis,:] = _get_clip_coords(diff_start + shift, diff_end + shift, cell_shape)
    
    tot_sq_diff = np.zeros((wfs_data.shape[0], n_pts_per_cell))
    buffer_2nd_diff = np.zeros((wfs_data.shape[0], n_pts_per_cell,3))
    tot_sq_2nd_diff = np.zeros((wfs_data.shape[0], n_pts_per_cell))
    tot_sq_val  = np.zeros(n_pts_per_cell)
    for cell_idx in range(cells_per_xsf):
        tot_sq_diff[:,diff_coords[0,:]] += (wfs_data_reorganized[:,(diff_coords[1,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)] -
                                            wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)])**2
        tot_sq_diff[:,diff_coords[0,:]] += (wfs_data_reorganized[:,(diff_coords[2,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)] -
                                            wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)])**2
        tot_sq_diff[:,diff_coords[0,:]] += (wfs_data_reorganized[:,(diff_coords[3,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)] -
                                            wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)])**2
        buffer_2nd_diff[:,diff_coords[0,:],0] += (wfs_data_reorganized[:,(diff_coords[1,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2 -
                                          wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2)
        buffer_2nd_diff[:,diff_coords[0,:],1] += (wfs_data_reorganized[:,(diff_coords[2,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2 -
                                          wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2)
        buffer_2nd_diff[:,diff_coords[0,:],2] += (wfs_data_reorganized[:,(diff_coords[3,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2 -
                                            wfs_data_reorganized[:,(diff_coords[0,:]+cell_idx*n_pts_per_cell)%(cells_per_xsf*n_pts_per_cell)]**2)

        tot_sq_2nd_diff[:,diff_coords[4,:]] += (buffer_2nd_diff[:,diff_coords[1,:]%n_pts_per_cell,0] -
                                                buffer_2nd_diff[:,diff_coords[0,:]%n_pts_per_cell,0])
        tot_sq_2nd_diff[:,diff_coords[4,:]] += (buffer_2nd_diff[:,diff_coords[2,:]%n_pts_per_cell,1] -
                                                buffer_2nd_diff[:,diff_coords[0,:]%n_pts_per_cell,1])
        tot_sq_2nd_diff[:,diff_coords[4,:]] += (buffer_2nd_diff[:,diff_coords[3,:]%n_pts_per_cell,2] -
                                                buffer_2nd_diff[:,diff_coords[0,:]%n_pts_per_cell,2])
        tot_sq_val += np.sum(wfs_data_reorganized[:,cell_idx*n_pts_per_cell : (cell_idx+1)*n_pts_per_cell]**2, axis=0)
    
    tot_sq_diff = tot_sq_diff.sum(axis=0)
    tot_sq_2nd_diff = abs(tot_sq_2nd_diff).sum(axis=0)
    print("done!")
    
    print("Sampling points for tetra-mesh: ", end='')
    sampling_weights = (sampling_prob_sq_diff*tot_sq_diff+
                        sampling_prob_sq_val*tot_sq_val+
                        sampling_prob_2nd_diff*tot_sq_2nd_diff**(exponent_2nd_diff)
    )
    sampling_weights += sampling_prob_baseline*max(sampling_weights)
    sampling_weights = (sampling_weights * np.ones(n_cells_in_domain).reshape((-1,1))).reshape(-1)
    sampling_weights /= sum(sampling_weights)
    # assert(sampling_weights.shape[0] == n_cells_in_domain * n_pts_per_cell)

    with mp.Pool() as pool:
        sampled_points = np.array(
            pool.map(_sampling_helper, [(sampling_weights[n_pts_per_cell*idx:n_pts_per_cell*(idx+1)],samples_per_cell, idx) for idx in range(n_cells_in_domain)])
        ).reshape(-1)
    # sampled_points = np.random.choice(np.arange(sampling_weights.shape[0]), samples_per_cell*n_cells_in_domain, p=sampling_weights, replace=False)
    shift_vec = wannier_cell[:,propagation_axis]*shape[propagation_axis]/(shape[propagation_axis]-1)/cells_per_xsf
    
    print("shift vec: ", shift_vec)
    n_overlap = int((cells_per_xsf-1)/2)
    pt_idxs_unit_cell = _get_clip_coords(n_overlap*coord_shift, (clip_end-clip_start-n_overlap*coord_shift).astype(np.int32),clip_end-clip_start)
    unitcell_points = grid.points[clipping_mask][pt_idxs_unit_cell]

    sampled_coords = (sampled_points//n_pts_per_cell).reshape((-1,1))*shift_vec + unitcell_points[sampled_points%n_pts_per_cell]
    wfs_data_big_grid = np.zeros((n_wfs*n_cells_in_domain, (2*n_overlap + n_cells_in_domain) * n_pts_per_cell))
    for wf_idx in range(n_wfs*n_cells_in_domain):
        wfs_data_big_grid[wf_idx, n_pts_per_cell*(wf_idx//n_wfs):n_pts_per_cell*(wf_idx//n_wfs + cells_per_xsf)] = wfs_data_reorganized[wf_idx%n_wfs,:]
    
    wfs_data_big_grid = wfs_data_big_grid[:,n_overlap*n_pts_per_cell:-n_overlap*n_pts_per_cell]
    print("done!")
    
    print("Generating mesh: ", end='')
    point_cloud = pv.PolyData(sampled_coords)
    tetra_mesh = point_cloud.delaunay_3d()

    cells = tetra_mesh.cells.reshape((-1,5))[:,1:]
    is_in_mesh = np.zeros(sampled_coords.shape[0], dtype=np.int8)
    is_in_mesh[cells.reshape(-1)] = 1
    sampled_coords = sampled_coords[is_in_mesh == 1, :]
    sampled_points = sampled_points[is_in_mesh == 1]
    print(sampled_coords.shape)

    point_cloud = pv.PolyData(sampled_coords)
    tetra_mesh = point_cloud.delaunay_3d()
    
    wfs_data_on_mesh = wfs_data_big_grid[:,sampled_points]
    print("done!")
    
    wf_sq_ints = (wfs_data_reorganized**2).sum(axis=1)*np.prod(unitcell_points[cell_shape[0]*cell_shape[1] + cell_shape[0] + 1]-unitcell_points[0])
    
    return tetra_mesh, wfs_data_on_mesh, tot_sq_diff, tot_sq_val, sampled_points%n_pts_per_cell, wf_sq_ints, tot_sq_2nd_diff, shift_vec

# the below function is very specific to the cnt example
def set_boundaries(mesh, V1, V2):
    surface_points = mesh.extract_surface().points
    surface_cells = mesh.extract_surface().faces.reshape((-1,4))[:,1:]
    surface_normals = mesh.extract_surface().compute_normals()['Normals']
    surface_point_indices_in_original_mesh = mesh.surface_indices()
    min_x = min(surface_points[:,0])
    max_x = max(surface_points[:,0])
    is_dirichlet = np.zeros(mesh.points.shape[0], dtype=np.int8)
    dirichlet_vals = np.zeros(mesh.points.shape[0])
    for cell_idx in range(surface_cells.shape[0]):
        if np.argmax(abs(surface_normals[cell_idx, :])) == 0:
            is_dirichlet[surface_point_indices_in_original_mesh[surface_cells[cell_idx]]] = 1
            if(max(surface_points[surface_cells[cell_idx],0])) < (min_x + max_x)/2:
                dirichlet_vals[surface_point_indices_in_original_mesh[surface_cells[cell_idx]]] = V1
            else:
                dirichlet_vals[surface_point_indices_in_original_mesh[surface_cells[cell_idx]]] = V2
    
    return (is_dirichlet, dirichlet_vals)

def _sampling_helper(args):
    weights, nsamples, cell_idx = args
    return np.random.choice(np.arange(weights.shape[0])+weights.shape[0]*cell_idx, nsamples, p=weights/sum(weights), replace=False)