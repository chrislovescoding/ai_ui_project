# src/data_generation/trajectory_generators.py
import random
import math

# --- Helper Functions ---
def linear_interpolate_pos(p0, p1, t):
    """Interpolates between two 2D points p0 and p1 at fraction t."""
    return (p0[0] * (1 - t) + p1[0] * t,
            p0[1] * (1 - t) + p1[1] * t)

def get_random_offscreen_pos(config, margin=20):
    """Gets a random position just off-screen."""
    side = random.choice(['top', 'bottom', 'left', 'right'])
    if side == 'top':
        return (random.uniform(0, config.WINDOW_WIDTH), -margin)
    elif side == 'bottom':
        return (random.uniform(0, config.WINDOW_WIDTH), config.WINDOW_HEIGHT + margin)
    elif side == 'left':
        return (-margin, random.uniform(0, config.WINDOW_HEIGHT))
    else: # right
        return (config.WINDOW_WIDTH + margin, random.uniform(0, config.WINDOW_HEIGHT))

def get_button_center(config):
    """Gets the center coordinates of the button."""
    bx, by, bw, bh = config.BUTTON_RECT_XYWH
    return (bx + bw / 2, by + bh / 2)

def get_random_point_on_button(config):
    """Gets a random point within the button's bounding box."""
    bx, by, bw, bh = config.BUTTON_RECT_XYWH
    return (random.uniform(bx, bx + bw), random.uniform(by, by + bh))

def get_random_onscreen_pos(config, exclude_button=False):
    """Gets a random position on screen, optionally excluding button area."""
    while True:
        x = random.uniform(0, config.WINDOW_WIDTH -1) # -1 to keep fully within
        y = random.uniform(0, config.WINDOW_HEIGHT -1)
        if not exclude_button:
            return (x,y)
        
        btn_x, btn_y, btn_w, btn_h = config.BUTTON_RECT_XYWH
        if not (btn_x <= x < btn_x + btn_w and btn_y <= y < btn_y + btn_h):
            return (x,y)

# --- Core Trajectory Building Blocks ---

def _static_segment(pos, mouse_state, num_frames):
    for _ in range(num_frames):
        yield int(pos[0]), int(pos[1]), mouse_state

def _linear_move_segment(start_pos, end_pos, mouse_state, num_frames):
    if num_frames <= 0:
        return
    if num_frames == 1:
        yield int(end_pos[0]), int(end_pos[1]), mouse_state
        return
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        curr_x, curr_y = linear_interpolate_pos(start_pos, end_pos, t)
        yield int(curr_x), int(curr_y), mouse_state

def _jitter_segment(center_pos, mouse_state, num_frames, max_offset=3):
    for _ in range(num_frames):
        jitter_x = random.uniform(-max_offset, max_offset)
        jitter_y = random.uniform(-max_offset, max_offset)
        yield int(center_pos[0] + jitter_x), int(center_pos[1] + jitter_y), mouse_state

def _arc_move_segment(start_pos, end_pos, mouse_state, num_frames, bulge_factor=0.3):
    """Simple arc using a quadratic Bezier-like midpoint displacement."""
    if num_frames <= 0: return
    mid_point_linear = linear_interpolate_pos(start_pos, end_pos, 0.5)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    bulge_dir = random.choice([-1, 1]) 
    control_point_offset_x = -dy * bulge_factor * bulge_dir
    control_point_offset_y = dx * bulge_factor * bulge_dir
    
    control_point = (mid_point_linear[0] + control_point_offset_x, 
                     mid_point_linear[1] + control_point_offset_y)

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        p0 = start_pos
        p1 = control_point
        p2 = end_pos
        
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        yield int(x), int(y), mouse_state

def _allocate_frames(total_frames, num_segments, min_frames_per_segment=1):
    """Distributes total_frames among num_segments, ensuring min_frames."""
    if num_segments <= 0: # Handle cases with no segments or invalid input
        if total_frames > 0: return [total_frames] # All frames to one implicit segment
        return []

    if total_frames < num_segments * min_frames_per_segment:
        # print(f"Warning: Not enough frames ({total_frames}) for {num_segments} segments with min {min_frames_per_segment}. Distributing proportionally.")
        avg_frames = total_frames // num_segments
        frames_alloc = [avg_frames] * num_segments
        remainder = total_frames % num_segments
        for i in range(remainder): frames_alloc[i] += 1
        return frames_alloc

    alloc = [min_frames_per_segment] * num_segments
    remaining_frames = total_frames - sum(alloc)
    # Distribute remaining frames randomly
    for _ in range(remaining_frames):
        alloc[random.randrange(num_segments)] += 1
    return alloc

# --- Full Scenario Trajectory Generators ---

# --- A. Simple Approaches & Departures ---
def linear_approach_hover_depart(config): # Covers A1, A2, B1
    total_frames = config.FRAMES_PER_EPISODE
    frame_allocs = _allocate_frames(total_frames, 3, min_frames_per_segment=max(1, total_frames // 10)) 
    start_pos = get_random_offscreen_pos(config)
    button_target = get_random_point_on_button(config)
    end_pos = get_random_offscreen_pos(config)
    while math.dist(start_pos, end_pos) < config.WINDOW_WIDTH / 2: end_pos = get_random_offscreen_pos(config)
    yield from _linear_move_segment(start_pos, button_target, "UP", frame_allocs[0])
    yield from _static_segment(button_target, "UP", frame_allocs[1])
    yield from _linear_move_segment(button_target, end_pos, "UP", frame_allocs[2])

def sweep_across_button(config): # Covers A3
    total_frames = config.FRAMES_PER_EPISODE
    if random.choice([True, False]): # Horizontal sweep
        start_y = random.uniform(config.BUTTON_RECT_XYWH[1] - 10, config.BUTTON_RECT_XYWH[1] + config.BUTTON_RECT_XYWH[3] + 10)
        # Ensure start_pos[0] and end_pos[0] are on opposite sides of the screen
        x_coords = sorted([get_random_offscreen_pos(config, margin=config.WINDOW_WIDTH*0.1)[0], 
                           -get_random_offscreen_pos(config, margin=config.WINDOW_WIDTH*0.1)[0]]) # Ensure different signs for x for sweep
        start_pos = (x_coords[0], start_y) 
        end_pos = (x_coords[1] if x_coords[1] != x_coords[0] else x_coords[1] + config.WINDOW_WIDTH + 40 , start_y) 
    else: # Vertical sweep
        start_x = random.uniform(config.BUTTON_RECT_XYWH[0] - 10, config.BUTTON_RECT_XYWH[0] + config.BUTTON_RECT_XYWH[2] + 10)
        y_coords = sorted([get_random_offscreen_pos(config, margin=config.WINDOW_HEIGHT*0.1)[1],
                           -get_random_offscreen_pos(config, margin=config.WINDOW_HEIGHT*0.1)[1]])
        start_pos = (start_x, y_coords[0])
        end_pos = (start_x, y_coords[1] if y_coords[1] != y_coords[0] else y_coords[1] + config.WINDOW_HEIGHT + 40)
    yield from _linear_move_segment(start_pos, end_pos, "UP", total_frames)


def precision_stop_at_edge(config): # Covers A4
    total_frames = config.FRAMES_PER_EPISODE
    btn_x, btn_y, btn_w, btn_h = config.BUTTON_RECT_XYWH
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    stop_type = random.choice(['outside', 'on_edge', 'inside'])
    # Offset for cursor hotspot relative to edge. 0 means hotspot is on the button's boundary line.
    # +ve is inside, -ve is outside.
    if stop_type == 'outside': offset = - (config.loaded_assets.cursor_width // 2 + 2) # Ensure sprite is fully outside
    elif stop_type == 'on_edge': offset = 0 # Hotspot on edge
    else: offset = config.loaded_assets.cursor_width // 2 + 2 # Ensure sprite is fully inside
    
    if edge == 'left': target_x,target_y,start_pos = btn_x+offset,random.uniform(btn_y,btn_y+btn_h),(btn_x+offset-random.uniform(50,100),random.uniform(btn_y,btn_y+btn_h))
    elif edge == 'right': target_x,target_y,start_pos = btn_x+btn_w+offset,random.uniform(btn_y,btn_y+btn_h),(btn_x+btn_w+offset+random.uniform(50,100),random.uniform(btn_y,btn_y+btn_h))
    elif edge == 'top': target_x,target_y,start_pos = random.uniform(btn_x,btn_x+btn_w),btn_y+offset,(random.uniform(btn_x,btn_x+btn_w),btn_y+offset-random.uniform(50,100))
    else: target_x,target_y,start_pos = random.uniform(btn_x,btn_x+btn_w),btn_y+btn_h+offset,(random.uniform(btn_x,btn_x+btn_w),btn_y+btn_h+offset+random.uniform(50,100))
    
    target_pos=(target_x,target_y)
    move_frames = max(1,total_frames//2)
    static_frames = total_frames-move_frames
    yield from _linear_move_segment(start_pos,target_pos,"UP",move_frames)
    yield from _static_segment(target_pos,"UP",static_frames)

def arc_approach_hover_depart(config): # Covers A6
    total_frames = config.FRAMES_PER_EPISODE
    frame_allocs = _allocate_frames(total_frames, 3, min_frames_per_segment=max(1,total_frames//10))
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);end_pos=get_random_offscreen_pos(config)
    while math.dist(start_pos,end_pos)<config.WINDOW_WIDTH/2:end_pos=get_random_offscreen_pos(config)
    bulge=random.uniform(0.2,0.5)
    yield from _arc_move_segment(start_pos,button_target,"UP",frame_allocs[0],bulge_factor=bulge)
    yield from _static_segment(button_target,"UP",frame_allocs[1])
    yield from _arc_move_segment(button_target,end_pos,"UP",frame_allocs[2],bulge_factor=bulge)

# --- B. Hovering & Clicking ---
def simple_click_on_button(config): # B2
    total_frames=config.FRAMES_PER_EPISODE; frame_allocs=_allocate_frames(total_frames,5,min_frames_per_segment=1)
    # Ensure press is short for a "simple click"
    frame_allocs[2]=random.randint(1,max(1,min(3,frame_allocs[2]))) # Press for 1-3 frames if possible
    
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);end_pos=get_random_offscreen_pos(config)
    while math.dist(start_pos,end_pos)<config.WINDOW_WIDTH/2:end_pos=get_random_offscreen_pos(config)
    yield from _linear_move_segment(start_pos,button_target,"UP",frame_allocs[0]) # Approach
    yield from _static_segment(button_target,"UP",frame_allocs[1])                # Hover Pre-click
    yield from _static_segment(button_target,"DOWN",frame_allocs[2])              # Press
    yield from _static_segment(button_target,"UP",frame_allocs[3])                # Release (Hover Post-click)
    yield from _linear_move_segment(button_target,end_pos,"UP",frame_allocs[4])   # Depart

def click_hold_release_on_button(config): # B3
    total_frames=config.FRAMES_PER_EPISODE;frame_allocs=_allocate_frames(total_frames,4,min_frames_per_segment=1)
    # Ensure PressHold is significant
    min_hold=max(1, total_frames//4)
    if frame_allocs[1] < min_hold : # If allocation gave too few frames for hold
        diff = min_hold - frame_allocs[1]
        frame_allocs[1] = min_hold
        # Steal frames from other segments (e.g., depart or approach)
        if frame_allocs[3] > diff + 1 : frame_allocs[3] -= diff
        elif frame_allocs[0] > diff + 1 : frame_allocs[0] -= diff
        # This readjustment is basic; a more robust one would re-run _allocate_frames with constraints.
            
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);end_pos=get_random_offscreen_pos(config)
    yield from _linear_move_segment(start_pos,button_target,"UP",frame_allocs[0]) # Approach
    yield from _static_segment(button_target,"DOWN",frame_allocs[1])              # Press & Hold
    yield from _static_segment(button_target,"UP",frame_allocs[2])                # Release (becomes Hover)
    yield from _linear_move_segment(button_target,end_pos,"UP",frame_allocs[3])    # Depart

# --- C. Drag Operations ---
def click_drag_off_release_off(config): # C1
    total_frames=config.FRAMES_PER_EPISODE;frame_allocs=_allocate_frames(total_frames,5,min_frames_per_segment=1)
    start_pos=get_random_offscreen_pos(config);button_press_target=get_random_point_on_button(config);drag_off_target=get_random_onscreen_pos(config,exclude_button=True)
    yield from _linear_move_segment(start_pos,button_press_target,"UP",frame_allocs[0]) # Approach
    yield from _static_segment(button_press_target,"DOWN",frame_allocs[1])               # Press briefly
    yield from _linear_move_segment(button_press_target,drag_off_target,"DOWN",frame_allocs[2]) # Drag Off
    yield from _static_segment(drag_off_target,"DOWN",frame_allocs[3])                   # Hold Off-button (mouse still down)
    yield from _static_segment(drag_off_target,"UP",frame_allocs[4])                     # Release Off-button

def click_drag_off_on_release_on(config): # C2
    total_frames=config.FRAMES_PER_EPISODE;frame_allocs=_allocate_frames(total_frames,6,min_frames_per_segment=1)
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);drag_intermediate_off_button=get_random_onscreen_pos(config,exclude_button=True);end_pos=get_random_offscreen_pos(config)
    yield from _linear_move_segment(start_pos,button_target,"UP",frame_allocs[0]) # Approach
    yield from _static_segment(button_target,"DOWN",frame_allocs[1])               # Press
    yield from _linear_move_segment(button_target,drag_intermediate_off_button,"DOWN",frame_allocs[2]) # Drag Off
    yield from _linear_move_segment(drag_intermediate_off_button,button_target,"DOWN",frame_allocs[3]) # Drag Back On
    yield from _static_segment(button_target,"UP",frame_allocs[4])                # Release on button (becomes Hover)
    yield from _linear_move_segment(button_target,end_pos,"UP",frame_allocs[5])    # Depart

# --- D. Jitter & Complex Paths ---
def jitter_on_button_then_click(config): # D1, D2
    total_frames=config.FRAMES_PER_EPISODE;frame_allocs=_allocate_frames(total_frames,6,min_frames_per_segment=max(1,total_frames//15));frame_allocs[2]=random.randint(1,max(1,min(3,frame_allocs[2])))
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);end_pos=get_random_offscreen_pos(config);jitter_amount=random.uniform(1,5)
    yield from _linear_move_segment(start_pos,button_target,"UP",frame_allocs[0])
    yield from _jitter_segment(button_target,"UP",frame_allocs[1],max_offset=jitter_amount)
    yield from _static_segment(button_target,"DOWN",frame_allocs[2]) # Static press for clarity
    yield from _jitter_segment(button_target,"DOWN",frame_allocs[3],max_offset=jitter_amount)
    yield from _jitter_segment(button_target,"UP",frame_allocs[4],max_offset=jitter_amount) # Jitter hover after release
    yield from _linear_move_segment(button_target,end_pos,"UP",frame_allocs[5])

def figure_eight_over_button(config): # D2
    total_frames=config.FRAMES_PER_EPISODE;btn_center_x,btn_center_y=get_button_center(config);radius_x=config.BUTTON_RECT_XYWH[2]*random.uniform(0.7,1.5);radius_y=config.BUTTON_RECT_XYWH[3]*random.uniform(0.7,1.5);start_offset_angle=random.uniform(0,2*math.pi)
    mouse_pattern=random.choice([["UP"],["UP","DOWN","UP"],["DOWN"]]);mouse_state_idx=0;
    frames_per_mouse_state = (total_frames // len(mouse_pattern)) if mouse_pattern and len(mouse_pattern) > 0 else total_frames
    current_frames_in_mouse_state=0;current_mouse_state=mouse_pattern[0] if mouse_pattern else "UP"
    for i in range(total_frames):
        if len(mouse_pattern) > 1 and current_frames_in_mouse_state>=frames_per_mouse_state :
            mouse_state_idx=(mouse_state_idx+1)%len(mouse_pattern);current_mouse_state=mouse_pattern[mouse_state_idx];current_frames_in_mouse_state=0
        angle=start_offset_angle+4*math.pi*(i/max(1,total_frames-1)); # 0 to 4pi for full figure eight
        x_offset=radius_x*math.sin(angle); y_offset=radius_y*math.sin(angle/2) # Lissajous for figure 8
        x=btn_center_x+x_offset; y=btn_center_y+y_offset
        yield int(x),int(y),current_mouse_state;current_frames_in_mouse_state+=1

# --- NEW: Fast Scribbling --- D3 variant
def fast_scribble_over_button_area(config):
    total_frames = config.FRAMES_PER_EPISODE
    button_center_x, button_center_y = get_button_center(config)
    # Scribble area slightly larger than button
    scribble_radius_w = config.BUTTON_RECT_XYWH[2] * random.uniform(0.7, 1.2) 
    scribble_radius_h = config.BUTTON_RECT_XYWH[3] * random.uniform(0.7, 1.2)
    
    num_segments = random.randint(max(3, total_frames // 10), max(5, total_frames // 4)) # More segments for "fast"
    
    # Calculate frames per segment more carefully to sum to total_frames
    base_frames_per_segment = total_frames // num_segments if num_segments > 0 else total_frames
    remainder_frames = total_frames % num_segments if num_segments > 0 else 0
    segment_frame_counts = [base_frames_per_segment + (1 if i < remainder_frames else 0) for i in range(num_segments)]
    if not segment_frame_counts and total_frames > 0: segment_frame_counts = [total_frames]


    current_pos = get_random_point_on_button(config) # Start scribble on/near button
    
    mouse_pattern = random.choice([["UP"], ["DOWN"], ["UP", "DOWN"]])
    mouse_state_idx = 0
    # Change mouse state less frequently than segments to allow some sustained UP/DOWN scribbles
    mouse_state_change_interval = total_frames // (len(mouse_pattern) * random.randint(1,2)) if mouse_pattern and len(mouse_pattern)>0 else total_frames 
    frames_in_current_mouse_state = 0
    current_mouse_state = mouse_pattern[0] if mouse_pattern else "UP"

    total_yielded_frames = 0
    for seg_idx, segment_frames in enumerate(segment_frame_counts):
        if segment_frames <= 0: continue

        target_x = button_center_x + random.uniform(-scribble_radius_w, scribble_radius_w)
        target_y = button_center_y + random.uniform(-scribble_radius_h, scribble_radius_h)
        target_pos = (max(0, min(config.WINDOW_WIDTH - 1, target_x)),
                      max(0, min(config.WINDOW_HEIGHT - 1, target_y)))
        
        for i in range(segment_frames):
            if len(mouse_pattern) > 1 and frames_in_current_mouse_state >= mouse_state_change_interval and mouse_state_change_interval > 0 :
                mouse_state_idx = (mouse_state_idx + 1) % len(mouse_pattern)
                current_mouse_state = mouse_pattern[mouse_state_idx]
                frames_in_current_mouse_state = 0
            
            t = i / max(1, segment_frames - 1) if segment_frames > 1 else 0 # Avoid div by zero for 1-frame segment
            pos_x, pos_y = linear_interpolate_pos(current_pos, target_pos, t)
            yield int(pos_x), int(pos_y), current_mouse_state
            frames_in_current_mouse_state += 1
            total_yielded_frames += 1
        current_pos = target_pos
    
    # Ensure exactly total_frames are yielded
    while total_yielded_frames < total_frames:
        yield int(current_pos[0]), int(current_pos[1]), current_mouse_state
        total_yielded_frames +=1


# --- E. Idle & Static States ---
def idle_off_button(config): # E1
    pos=get_random_onscreen_pos(config,exclude_button=True);mouse_state=random.choice(["UP","DOWN"])
    yield from _static_segment(pos,mouse_state,config.FRAMES_PER_EPISODE)
def idle_on_button_hover(config): # E2
    pos=get_random_point_on_button(config)
    yield from _static_segment(pos,"UP",config.FRAMES_PER_EPISODE)
def idle_on_button_pressed(config): # E3
    pos=get_random_point_on_button(config)
    yield from _static_segment(pos,"DOWN",config.FRAMES_PER_EPISODE)

# --- F. Boundary & Off-Screen Conditions ---
def pixel_scan_button_edge(config): # F3
    total_frames = config.FRAMES_PER_EPISODE
    btn_x, btn_y, btn_w, btn_h = config.BUTTON_RECT_XYWH
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    
    # Scan for a distance covering cursor width plus some margin
    scan_distance = config.loaded_assets.cursor_width + random.randint(10, 30) 
    
    if total_frames < scan_distance : scan_distance = total_frames # Cap scan if not enough frames
    if scan_distance <= 0: # Not enough frames to scan even 1 pixel
        yield from idle_off_button(config) # Fallback to a simple static trajectory
        return

    frames_per_pixel_step = total_frames // scan_distance
    extra_hold_frames = total_frames % scan_distance # Hold last position

    # Start scan from outside the button, scan across the edge, and end up inside/other side
    start_scan_offset = - (scan_distance // 2) 

    current_x, current_y = 0, 0 # Will be updated in loops

    if edge == 'left':
        fixed_coord = btn_y + btn_h / 2 # Fixed y for horizontal scan
        scan_start_coord = btn_x + start_scan_offset
        for i in range(scan_distance):
            current_x = scan_start_coord + i; current_y = fixed_coord
            for _ in range(frames_per_pixel_step): yield int(current_x), int(current_y), "UP"
    elif edge == 'right':
        fixed_coord = btn_y + btn_h / 2
        scan_start_coord = btn_x + btn_w + start_scan_offset
        for i in range(scan_distance):
            current_x = scan_start_coord + i; current_y = fixed_coord
            for _ in range(frames_per_pixel_step): yield int(current_x), int(current_y), "UP"
    elif edge == 'top':
        fixed_coord = btn_x + btn_w / 2 # Fixed x for vertical scan
        scan_start_coord = btn_y + start_scan_offset
        for i in range(scan_distance):
            current_x = fixed_coord; current_y = scan_start_coord + i
            for _ in range(frames_per_pixel_step): yield int(current_x), int(current_y), "UP"
    else: # bottom
        fixed_coord = btn_x + btn_w / 2
        scan_start_coord = btn_y + btn_h + start_scan_offset
        for i in range(scan_distance):
            current_x = fixed_coord; current_y = scan_start_coord + i
            for _ in range(frames_per_pixel_step): yield int(current_x), int(current_y), "UP"
            
    # Hold last position for any remaining frames
    for _ in range(extra_hold_frames):
         yield int(current_x), int(current_y), "UP"

# --- NEW: Spiral Trajectory --- (Mathematical Path)
def spiral_around_button(config):
    total_frames = config.FRAMES_PER_EPISODE
    center_x, center_y = get_button_center(config)
    
    start_radius = random.uniform(config.BUTTON_RECT_XYWH[2]*0.1, config.BUTTON_RECT_XYWH[2]*0.5) 
    end_radius = random.uniform(config.WINDOW_WIDTH*0.4, config.WINDOW_WIDTH*0.7) 
    if random.choice([True, False]): start_radius, end_radius = end_radius, start_radius # Spiral in or out

    num_rotations = random.uniform(1.5, 4.5) 
    start_angle_offset = random.uniform(0, 2 * math.pi) # Start spiral from random angle

    mouse_pattern = random.choice([["UP"], ["UP", "DOWN"], ["DOWN", "UP"]])
    mouse_state_idx = 0
    frames_per_mouse_state = (total_frames // len(mouse_pattern)) if mouse_pattern and len(mouse_pattern) > 0 else total_frames
    current_frames_in_mouse_state = 0
    current_mouse_state = mouse_pattern[0] if mouse_pattern else "UP"

    for i in range(total_frames):
        if len(mouse_pattern) > 1 and frames_per_mouse_state > 0 and current_frames_in_mouse_state >= frames_per_mouse_state :
            mouse_state_idx = (mouse_state_idx + 1) % len(mouse_pattern)
            current_mouse_state = mouse_pattern[mouse_state_idx]
            current_frames_in_mouse_state = 0
        
        t = i / max(1, total_frames - 1) 
        current_radius = start_radius * (1 - t) + end_radius * t
        current_angle = start_angle_offset + (t * num_rotations * 2 * math.pi)

        x = center_x + current_radius * math.cos(current_angle)
        y = center_y + current_radius * math.sin(current_angle)
        
        x = max(0, min(config.WINDOW_WIDTH - 1, x))
        y = max(0, min(config.WINDOW_HEIGHT - 1, y))

        yield int(x), int(y), current_mouse_state
        current_frames_in_mouse_state += 1


# --- NEW: Frame-Perfect Synchronized Event --- (G5 variant)
def synchronized_press_on_enter(config):
    total_frames = config.FRAMES_PER_EPISODE
    if total_frames < 3: # Needs at least 3 frames for approach, event, depart
        yield from idle_on_button_hover(config); return

    # Aim for synchronized event around the middle of the episode
    sync_frame_target_idx = random.randint(max(1, total_frames // 3), max(1, total_frames - (total_frames // 3) -1))
    
    frames_approach = sync_frame_target_idx
    frames_hold_pressed = random.randint(1, max(1, min(5, total_frames // 8))) # Short hold after sync event
    frames_depart = total_frames - frames_approach - frames_hold_pressed
    
    # Ensure depart has at least one frame. Steal from hold if necessary.
    if frames_depart <= 0:
        needed_for_depart = 1 - frames_depart
        frames_hold_pressed = max(1, frames_hold_pressed - needed_for_depart)
        frames_depart = 1
    # Ensure approach has at least one frame. Harder to adjust this one without recalculating sync_frame_target_idx.
    # This setup assumes frames_approach (sync_frame_target_idx) is chosen well.
    if frames_approach <=0: frames_approach = 1 # Should not happen withrandint above

    btn_x, btn_y, btn_w, btn_h = config.BUTTON_RECT_XYWH
    
    # Point where cursor hotspot will be when mouse goes DOWN.
    # Choose a random point on one of the button edges.
    edge_choice = random.choice(['left', 'right', 'top', 'bottom'])
    if edge_choice == 'left':   enter_pos = (btn_x, random.uniform(btn_y, btn_y + btn_h))
    elif edge_choice == 'right':  enter_pos = (btn_x + btn_w, random.uniform(btn_y, btn_y + btn_h))
    elif edge_choice == 'top':    enter_pos = (random.uniform(btn_x, btn_x + btn_w), btn_y)
    else: # bottom
        enter_pos = (random.uniform(btn_x, btn_x + btn_w), btn_y + btn_h)

    # Start position for approach, ensuring it's off-button before the sync frame.
    # Move from a point such that at t=1 of approach, cursor is at enter_pos.
    # Start slightly further away to ensure it's off-button.
    start_pos_offset_factor = 1.2 # Start 20% further than the approach segment length would imply
    if edge_choice == 'left':   start_pos = (enter_pos[0] - (config.loaded_assets.cursor_width * start_pos_offset_factor), enter_pos[1])
    elif edge_choice == 'right':  start_pos = (enter_pos[0] + (config.loaded_assets.cursor_width * start_pos_offset_factor), enter_pos[1])
    elif edge_choice == 'top':    start_pos = (enter_pos[0], enter_pos[1] - (config.loaded_assets.cursor_height * start_pos_offset_factor))
    else: # bottom
        start_pos = (enter_pos[0], enter_pos[1] + (config.loaded_assets.cursor_height * start_pos_offset_factor))

    # 1. Approach (Yields frames_approach frames, mouse UP)
    # The last point of this segment will be *just before* enter_pos
    for i in range(frames_approach):
        # t goes from 0 to (frames_approach-1)/frames_approach.
        # We want t to be 1 at the *end* of the last frame *of approach*
        # to land exactly on enter_pos for the *next* (sync) frame.
        t = (i + 1) / (frames_approach + 1) # t for interpolation up to the sync point
        curr_x, curr_y = linear_interpolate_pos(start_pos, enter_pos, t)
        yield int(curr_x), int(curr_y), "UP"
        
    # 2. Synchronized Event Frame (at enter_pos, mouse DOWN) + Hold Pressed
    for _ in range(frames_hold_pressed):
        yield int(enter_pos[0]), int(enter_pos[1]), "DOWN"
        
    # 3. Depart from enter_pos
    end_pos_depart = get_random_offscreen_pos(config)
    mouse_state_depart = random.choice(["UP", "DOWN"]) # If DOWN, it's a drag off from the pressed state
    yield from _linear_move_segment(enter_pos, end_pos_depart, mouse_state_depart, frames_depart)


# --- G. Rapid Actions & Stress Tests ---
def teleport_onto_off_button(config): # G1
    total_frames=config.FRAMES_PER_EPISODE;
    if total_frames < 3: yield from idle_off_button(config); return # Needs at least 3 distinct phases
    
    frame_allocs = _allocate_frames(total_frames, 3, min_frames_per_segment=1)
    frames_before_teleport = frame_allocs[0]
    frames_at_teleport_target = frame_allocs[1]
    frames_after_teleport = frame_allocs[2]
    
    start_pos=get_random_onscreen_pos(config,exclude_button=True);teleport_target=get_random_point_on_button(config);end_pos=get_random_onscreen_pos(config,exclude_button=True)
    if random.random()<0.5:start_pos,teleport_target=teleport_target,start_pos # Randomly teleport on or off
    mouse_state=random.choice(["UP","DOWN"])
    yield from _static_segment(start_pos,mouse_state,frames_before_teleport)
    yield from _static_segment(teleport_target,mouse_state,frames_at_teleport_target)
    yield from _static_segment(end_pos,mouse_state,frames_after_teleport)

def rapid_multiple_clicks(config): # G2
    total_frames=config.FRAMES_PER_EPISODE;
    num_clicks=random.randint(2,max(2, total_frames // 6)) # More clicks if more frames
    
    # Segments: Approach, (Press, Release)*N, Depart = 1 + 2N + 1
    min_total_segments= 2 + 2*num_clicks
    if total_frames < min_total_segments : # Not enough frames for this many clicks
        num_clicks = max(1, (total_frames - 2) // 2) # Reduce clicks
        min_total_segments = 2 + 2*num_clicks
    
    frame_allocs=_allocate_frames(total_frames,min_total_segments,min_frames_per_segment=1)
    
    start_pos=get_random_offscreen_pos(config);button_target=get_random_point_on_button(config);end_pos=get_random_offscreen_pos(config);idx=0
    
    yield from _linear_move_segment(start_pos,button_target,"UP",frame_allocs[idx]);idx+=1
    for _ in range(num_clicks):
        if idx + 1 >= len(frame_allocs): break # Safety break if allocation is off
        yield from _static_segment(button_target,"DOWN",frame_allocs[idx]);idx+=1 # Press
        yield from _static_segment(button_target,"UP",frame_allocs[idx]);idx+=1   # Release
    
    # Remaining frames for departure (if any left in allocs)
    if idx < len(frame_allocs):
        yield from _linear_move_segment(button_target,end_pos,"UP",frame_allocs[idx])
    elif total_frames - sum(frame_allocs[:idx]) > 0 : # If alloc was short but frames remain
         yield from _linear_move_segment(button_target,end_pos,"UP", total_frames - sum(frame_allocs[:idx]))


# --- H. No-Op / Null Scenarios ---
def movement_entirely_off_button(config): # H1
    total_frames=config.FRAMES_PER_EPISODE;start_pos=get_random_onscreen_pos(config,exclude_button=True);end_pos=get_random_onscreen_pos(config,exclude_button=True);mouse_state=random.choice(["UP","DOWN"])
    yield from _linear_move_segment(start_pos,end_pos,mouse_state,total_frames)


# --- List of available trajectory generators ---
AVAILABLE_TRAJECTORIES = {
    "everyday": [
        linear_approach_hover_depart,
        arc_approach_hover_depart,
        simple_click_on_button,
        click_hold_release_on_button,
        idle_off_button, 
        idle_on_button_hover,
        movement_entirely_off_button,
    ],
    "edge_case": [
        sweep_across_button, 
        precision_stop_at_edge,
        click_drag_off_release_off,
        click_drag_off_on_release_on,
        jitter_on_button_then_click,
        figure_eight_over_button, 
        idle_on_button_pressed, 
        pixel_scan_button_edge, 
        teleport_onto_off_button,
        rapid_multiple_clicks,
        # Added new ones
        spiral_around_button,
        synchronized_press_on_enter,
        fast_scribble_over_button_area,
    ]
}

# Ensure lists are not empty
if not AVAILABLE_TRAJECTORIES["everyday"]: AVAILABLE_TRAJECTORIES["everyday"].append(idle_off_button) 
if not AVAILABLE_TRAJECTORIES["edge_case"]: AVAILABLE_TRAJECTORIES["edge_case"].append(pixel_scan_button_edge)


if __name__ == '__main__':
    class DummyConfigAssets: 
        def __init__(self):
            self.cursor_width = 32 # Example cursor dim
            self.cursor_height = 32
    class DummyConfig:
        def __init__(self):
            self.FRAMES_PER_EPISODE = 30
            self.WINDOW_WIDTH = 256
            self.WINDOW_HEIGHT = 256
            self.BUTTON_RECT_XYWH = [78, 110, 100, 36]
            self.loaded_assets = DummyConfigAssets() # Attach dummy assets

    dummy_cfg = DummyConfig()
    print("Testing ALL trajectory generators...")
    all_funcs_tested = 0
    all_funcs_ok = 0
    for category, funcs in AVAILABLE_TRAJECTORIES.items():
        print(f"\n--- {category.upper()} TRAJECTORIES ---")
        if not funcs: print(" (No functions defined)"); continue
        for func in funcs:
            all_funcs_tested += 1
            print(f"  Testing {func.__name__}:")
            count = 0; error_occurred = False; first_frame_data=None; last_frame_data=None
            try:
                for x, y, mouse in func(dummy_cfg):
                    if count == 0: first_frame_data = (x,y,mouse)
                    last_frame_data = (x,y,mouse) # Keep updating last frame
                    count += 1
                
                if first_frame_data: print(f"    Frame 0: {first_frame_data}")
                if last_frame_data and count > 1 : print(f"    Frame {count-1}: {last_frame_data}")
                
                if count != dummy_cfg.FRAMES_PER_EPISODE:
                    print(f"    ERROR: {func.__name__} yielded {count} frames, expected {dummy_cfg.FRAMES_PER_EPISODE}")
                    error_occurred = True
                else:
                    print(f"    OK: Yielded {count} frames.")
                    all_funcs_ok +=1
            except Exception as e:
                print(f"    EXCEPTION in {func.__name__}: {e}")
                import traceback; traceback.print_exc()
                error_occurred = True
    print(f"\n--- Summary ---")
    print(f"Total functions tested: {all_funcs_tested}")
    print(f"Functions OK: {all_funcs_ok}")
    if all_funcs_tested != all_funcs_ok:
        print(f"Functions with errors: {all_funcs_tested - all_funcs_ok}")