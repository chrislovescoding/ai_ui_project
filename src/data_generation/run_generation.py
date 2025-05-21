# src/data_generation/run_generation.py
import os
import random
import time
import yaml
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .config import load_config_and_assets # Relative import
from .episode_generator import generate_episode_triplets # Relative import
from .trajectory_generators import AVAILABLE_TRAJECTORIES # Relative import

# Global config for worker processes. Must be populated in main.
# This is a common pattern for multiprocessing with shared, read-only data.
# Each worker process will re-initialize its own config and assets if passed by path.
# Alternatively, if config object is small and pickleable, it can be passed.
# For assets like loaded fonts/images, it's better for each worker to load them.
# Let's pass the config_path and let workers load.

def worker_process_episode(args):
    episode_id, trajectory_func_name_str, config_path = args
    
    # Each worker loads its own config and assets to avoid sharing complex objects
    # that might not be pickleable or safe across processes (like PIL ImageFont).
    try:
        # Find the trajectory function by its name string
        # This assumes trajectory_generators.py defines functions at the top level.
        # A more robust way might involve a registry if functions are in classes/nested.
        
        # This part is a bit tricky. How to get the function object from its name string
        # when trajectory_generators module isn't directly imported here in the worker?
        # Simplest: make trajectory_func_name_str point to one of the functions
        # in AVAILABLE_TRAJECTORIES which are actual function objects.
        # For this to work, trajectory_func_name_str should be the function object itself if pickleable,
        # or we reconstruct it.
        # The current AVAILABLE_TRAJECTORIES stores function objects, which should be fine.
        
        # Let's assume trajectory_func_name_str is actually the function *object*
        # made pickleable by being a top-level function in its module.
        
        config_obj = load_config_and_assets(config_path) # Worker loads its own
        trajectory_callable = trajectory_func_name_str # Assuming it's the func object

        generate_episode_triplets(episode_id, trajectory_callable, config_obj)
        return episode_id, None # Success
    except Exception as e:
        # Log error and return it for main process to see
        # print(f"Error in worker for episode {episode_id} with {trajectory_func_name_str.__name__}: {e}")
        # import traceback
        # traceback.print_exc()
        return episode_id, e # Failure

def main():
    start_time = time.time()
    config_path = "assets/design_spec.yaml" # Central place for config file path
    
    # Load config once in main process to get generation parameters
    main_config = load_config_and_assets(config_path)
    
    os.makedirs(main_config.OUTPUT_DIR_BASE, exist_ok=True)

    tasks = []
    current_episode_id = 0

    # Everyday episodes
    print(f"Preparing {main_config.NUM_EVERYDAY_EPISODES} everyday episodes...")
    for _ in range(main_config.NUM_EVERYDAY_EPISODES):
        if not AVAILABLE_TRAJECTORIES["everyday"]:
            print("Warning: No 'everyday' trajectories defined. Skipping.")
            break
        trajectory_func = random.choice(AVAILABLE_TRAJECTORIES["everyday"])
        tasks.append((current_episode_id, trajectory_func, config_path))
        current_episode_id += 1

    # Edge-case episodes
    print(f"Preparing {main_config.NUM_EDGE_CASE_EPISODES} edge-case episodes...")
    for _ in range(main_config.NUM_EDGE_CASE_EPISODES):
        if not AVAILABLE_TRAJECTORIES["edge_case"]:
            print("Warning: No 'edge_case' trajectories defined. Skipping.")
            break
        trajectory_func = random.choice(AVAILABLE_TRAJECTORIES["edge_case"])
        tasks.append((current_episode_id, trajectory_func, config_path))
        current_episode_id += 1
    
    if not tasks:
        print("No tasks generated. Exiting.")
        return

    print(f"Total tasks to process: {len(tasks)}")
    
    num_actual_workers = min(main_config.NUM_WORKERS, cpu_count(), len(tasks))
    if num_actual_workers < 1: num_actual_workers = 1
    
    print(f"Starting generation with {num_actual_workers} worker processes...")

    # Filter out tasks if trajectory_func is None (e.g. if lists were empty)
    valid_tasks = [task for task in tasks if task[1] is not None]
    if not valid_tasks:
        print("No valid tasks with trajectory functions. Exiting.")
        return

    failed_episodes = []
    with Pool(processes=num_actual_workers) as pool:
        # Using imap_unordered for potentially better performance with varying task times
        # tqdm provides the progress bar
        results = list(tqdm(pool.imap_unordered(worker_process_episode, valid_tasks), total=len(valid_tasks), desc="Generating Episodes"))
    
    for ep_id, error in results:
        if error:
            failed_episodes.append((ep_id, error))

    print("\n--- Generation Summary ---")
    print(f"Total episodes scheduled: {len(valid_tasks)}")
    successful_episodes = len(valid_tasks) - len(failed_episodes)
    print(f"Successfully generated: {successful_episodes}")
    if failed_episodes:
        print(f"Failed episodes: {len(failed_episodes)}")
        # for ep_id, err_msg in failed_episodes:
        #     print(f"  Ep {ep_id}: {err_msg}") # Potentially very verbose
    else:
        print("All episodes generated successfully!")

    # Create dataset.yaml manifest
    manifest_path = os.path.join(main_config.OUTPUT_DIR_BASE, "dataset.yaml")
    manifest_data = {
        "total_episodes_generated": successful_episodes, # Only count successful ones
        "num_everyday_episodes_scheduled": main_config.NUM_EVERYDAY_EPISODES,
        "num_edge_case_episodes_scheduled": main_config.NUM_EDGE_CASE_EPISODES,
        "frames_per_episode": main_config.FRAMES_PER_EPISODE, # This is number of states
        "triplets_per_episode": main_config.FRAMES_PER_EPISODE -1 if main_config.FRAMES_PER_EPISODE > 0 else 0,
        "window_size": [main_config.WINDOW_WIDTH, main_config.WINDOW_HEIGHT],
        "button_rect_xywh": main_config.BUTTON_RECT_XYWH,
        # Add more relevant design spec details from main_config.data if needed
        "generator_version": "1.0_initial", # Placeholder, use git hash in real project
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Initial dataset generated with basic trajectories."
    }
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest_data, f, indent=2, sort_keys=False)
    print(f"Dataset manifest saved to {manifest_path}")

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # To run this script, you would typically navigate to the `project_root` directory
    # and run: python -m src.data_generation.run_generation
    # This ensures relative imports within the src.data_generation package work.
    main()