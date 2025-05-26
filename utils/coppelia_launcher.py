# utils/coppelia_launcher.py
import os
import platform
import subprocess
import time

# --- Global state for the CoppeliaSim process handle ---
_coppeliasim_process_handle = None

# --- Configuration ---
# YOU MUST SET THIS PATH TO YOUR COPPELIASIM INSTALLATION DIRECTORY
COPPELIASIM_INSTALL_PATH = "C:\\Program Files\\CoppeliaRobotics\\CoppeliaSimEdu" # WINDOWS
# COPPELIASIM_INSTALL_PATH = "/opt/CoppeliaSim_Edu" # LINUX
# COPPELIASIM_INSTALL_PATH = "/Applications/CoppeliaSim.app" # MACOS

'''"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe" -h C:\Users\user\Dropbox\PC\Desktop\uni\Leiden\24S2\robotics\final\robotics_final\scenes\rex_camera.ttt'''

def _determine_paths(coppeliasim_root_path, scene_file_name):
    system = platform.system()
    executable_path = None
    
    if system == "Windows":
        executable_path = os.path.join(coppeliasim_root_path, "coppeliaSim.exe")
    elif system == "Linux":
        executable_path = os.path.join(coppeliasim_root_path, "coppeliaSim.sh")
    elif system == "Darwin": # macOS
        macos_exec_path = os.path.join(coppeliasim_root_path, "Contents", "MacOS", "coppeliaSim")
        if os.path.exists(macos_exec_path) and coppeliasim_root_path.endswith(".app"):
            executable_path = macos_exec_path
        else: 
            executable_path = os.path.join(coppeliasim_root_path, "coppeliaSim") 
    else:
        print(f"Unsupported OS: {system}")
        return None, None

    if not os.path.exists(executable_path):
        print(f"Error: CoppeliaSim executable not found at '{executable_path}'. Check COPPELIASIM_INSTALL_PATH.")
        return None, None

    # Scene path logic: Prioritize './scenes/' relative to CWD
    script_cwd = os.getcwd()
    path_in_script_scenes_folder = os.path.join(script_cwd, "scenes", scene_file_name)
    
    scene_to_load = None
    if os.path.exists(path_in_script_scenes_folder):
        scene_to_load = path_in_script_scenes_folder
    else: # Fallback checks (simplified here, can be expanded if needed)
        path_in_install_scenes = os.path.join(coppeliasim_root_path, "scenes", scene_file_name)
        if os.path.exists(path_in_install_scenes):
            scene_to_load = path_in_install_scenes
        else:
            path_in_install_root = os.path.join(coppeliasim_root_path, scene_file_name)
            if os.path.exists(path_in_install_root):
                 scene_to_load = path_in_install_root
            else:
                print(f"Warning: Scene '{scene_file_name}' not found in './scenes/' or CoppeliaSim install paths.")
    
    return executable_path, scene_to_load

def start_coppeliasim(scene_file_name, headless=False, port=19997):
    global _coppeliasim_process_handle
    if _coppeliasim_process_handle and _coppeliasim_process_handle.poll() is None:
        print("CoppeliaSim already launched by this script.")
        return True

    if not COPPELIASIM_INSTALL_PATH or not os.path.isdir(COPPELIASIM_INSTALL_PATH):
        print("Error: COPPELIASIM_INSTALL_PATH is not set or invalid in utils/coppelia_launcher.py.")
        return False

    executable_path, scene_path_arg = _determine_paths(COPPELIASIM_INSTALL_PATH, scene_file_name)
    if not executable_path:
        return False

    cmd = [executable_path]
    if headless:
        cmd.append("-h")
    
    # Optional: Command to auto-start ZMQ remote API if supported reliably by your version
    # cmd.append(f"-gREMOTEAPISERVERSERVICE_{port}_FALSE_TRUE")
    # For now, rely on the scene's internal script to start ZMQ.

    if scene_path_arg:
        cmd.append(scene_path_arg)
    else:
        print("Warning: No scene file path provided to CoppeliaSim launcher.")


    print(f"Launching CoppeliaSim: {' '.join(cmd)}")
    try:
        _coppeliasim_process_handle = subprocess.Popen(cmd)
        print(f"CoppeliaSim process started (PID: {_coppeliasim_process_handle.pid}). Waiting for init...")
        time.sleep(15) # Adjust wait time as needed
        if _coppeliasim_process_handle.poll() is not None:
            print("Error: CoppeliaSim process exited prematurely.")
            _coppeliasim_process_handle = None
            return False
        print("CoppeliaSim presumed initialized.")
        return True
    except Exception as e:
        print(f"Error launching CoppeliaSim: {e}")
        _coppeliasim_process_handle = None
        return False

def stop_coppeliasim():
    global _coppeliasim_process_handle
    if _coppeliasim_process_handle and _coppeliasim_process_handle.poll() is None:
        print(f"Stopping CoppeliaSim process (PID: {_coppeliasim_process_handle.pid})...")
        _coppeliasim_process_handle.terminate()
        try:
            _coppeliasim_process_handle.wait(timeout=5)
            print("CoppeliaSim process terminated.")
        except subprocess.TimeoutExpired:
            print("Coppeliasim timed out on terminate, killing.")
            _coppeliasim_process_handle.kill()
            print("CoppeliaSim process killed.")
        _coppeliasim_process_handle = None
    elif _coppeliasim_process_handle:
        print("CoppeliaSim process was already stopped.")
        _coppeliasim_process_handle = None
    else:
        print("No CoppeliaSim process was managed by this script to stop.")

if __name__ == '__main__': # For testing the launcher
    print("Testing CoppeliaSim Launcher...")
    if start_coppeliasim("rex_camera.ttt", headless=False): # Test with a scene
        print("CoppeliaSim should be running. Waiting 10s before stopping.")
        time.sleep(10)
    else:
        print("Failed to launch CoppeliaSim.")
    stop_coppeliasim()
    print("Launcher test finished.")