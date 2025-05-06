# env/scene_elements.py
import numpy as np
import random

def generate_random_mountain(
    sim,
    target_total_height=1.0,    # Default total height
    max_cylinder_height=0.4,    # Default max height for any single layer
    min_cylinder_height=0.05,   # Default min height for any single layer
    area_bounds_x=(-2.0, 2.0),  # Default X-range for mountain center
    area_bounds_y=(1.5, 3.0),   # Default Y-range (places it generally in front)
    peak_radius=0.1,            # Default radius of the top cylinder
    base_radius_factor=6.0,     # Default base radius = peak_radius * factor
    shape_options=8             # Default shape options (e.g., visible, respondable)
):
    """
    Generates a randomly placed mountain made of stacked cylinders.
    Cylinders are wider at the bottom and narrower at the top.
    Cylinder heights are generally taller at the bottom and shorter at the top.

    Args:
        sim: CoppeliaSim sim object.
        target_total_height (float): The desired total height of the mountain.
        max_cylinder_height (float): Max height for any single cylinder layer.
        min_cylinder_height (float): Min height for any single cylinder layer.
        area_bounds_x (tuple): (min_x, max_x) for placing the mountain's X center.
        area_bounds_y (tuple): (min_y, max_y) for placing the mountain's Y center.
        peak_radius (float): Radius of the top-most cylinder.
        base_radius_factor (float): Factor to determine base radius from peak radius.
        shape_options (int): Options for sim.createPrimitiveShape.

    Returns:
        list: A list of handles to the created cylinder objects.
    """
    handles = []

    # --- Parameter Validation and Sanitization ---
    if target_total_height < min_cylinder_height:
        target_total_height = min_cylinder_height
    if peak_radius <= 0.001: 
        peak_radius = 0.01
    if base_radius_factor < 1.0: 
        base_radius_factor = 1.0
    if min_cylinder_height <= 0.001:
        min_cylinder_height = 0.01
    if max_cylinder_height < min_cylinder_height:
        max_cylinder_height = min_cylinder_height
    
    # --- Mountain Placement ---
    mountain_center_x = random.uniform(area_bounds_x[0], area_bounds_x[1])
    mountain_center_y = random.uniform(area_bounds_y[0], area_bounds_y[1])

    layers_params = []  # Stores {'height': h, 'radius': r} for each layer, from bottom-to-top

    height_remaining = target_total_height
    previous_layer_h_for_decreasing_trend = max_cylinder_height + 1.0 

    while height_remaining >= min_cylinder_height - 1e-5 : 
        current_potential_max_h = min(max_cylinder_height, previous_layer_h_for_decreasing_trend)
        current_potential_min_h = min_cylinder_height
        
        if current_potential_min_h > current_potential_max_h : 
            cyl_h_candidate = current_potential_min_h
        else:
            cyl_h_candidate = random.uniform(current_potential_min_h, current_potential_max_h)

        actual_cyl_height = min(cyl_h_candidate, height_remaining)

        if (height_remaining - actual_cyl_height < min_cylinder_height) and \
           (height_remaining - actual_cyl_height > 1e-5): 
            actual_cyl_height = height_remaining
        
        if actual_cyl_height < min_cylinder_height - 1e-5: 
            if layers_params: 
                layers_params[-1]['height'] += actual_cyl_height
                height_remaining -= actual_cyl_height
            break 

        layers_params.append({'height': actual_cyl_height, 'radius': 0.0}) 
        height_remaining -= actual_cyl_height
        previous_layer_h_for_decreasing_trend = actual_cyl_height

    if not layers_params and target_total_height >= min_cylinder_height:
        layers_params.append({'height': target_total_height, 'radius': peak_radius})

    # --- Assign Radii ---
    num_layers = len(layers_params)
    if num_layers > 0:
        max_r = peak_radius * base_radius_factor 
        min_r = peak_radius                  

        if num_layers == 1:
            layers_params[0]['radius'] = (min_r + max_r) / 2.0 
        else:
            for i in range(num_layers):
                t = i / (num_layers - 1.0)
                current_radius = max_r * (1.0 - t) + min_r * t
                layers_params[i]['radius'] = max(current_radius, 0.01)

    # --- Create Cylinder Objects ---
    current_z_base = 0.0
    for layer_info in layers_params:
        cyl_h = layer_info['height']
        cyl_r = layer_info['radius']

        if cyl_h < 0.001 or cyl_r < 0.001: 
            continue

        cyl_center_z = current_z_base + cyl_h / 2.0
        try:
            cyl = sim.createPrimitiveShape(
                sim.primitiveshape_cylinder,
                [2 * cyl_r, 2 * cyl_r, cyl_h],
                shape_options
            )
            sim.setObjectPosition(cyl, -1, [mountain_center_x, mountain_center_y, cyl_center_z])
            sim.setObjectOrientation(cyl, -1, [0, 0, 0])
            sim.setObjectInt32Param(cyl, sim.shapeintparam_static, 1)
            sim.setObjectInt32Param(cyl, sim.shapeintparam_respondable, 1)
            handles.append(cyl)
        except Exception as e:
            print(f"Error creating mountain layer (H:{cyl_h:.2f}, R:{cyl_r:.2f}, Z_base:{current_z_base:.2f}): {e}")

        current_z_base += cyl_h

    return handles
