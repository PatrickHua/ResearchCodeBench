import subprocess
import concurrent.futures

def run_shell_command(command):
    """
    Run a shell command and return success status, exit code, and output.
    
    Args:
        command: The shell command to run
        
    Returns:
        tuple: (success, exit_code, stdout, stderr)
    """
    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            shell=True,
            check=False,  # Don't raise exception on non-zero exit
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Return strings instead of bytes
        )
        
        # Get the exit code and output
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        
        # Check if the command succeeded
        success = exit_code == 0
        
        return success, exit_code, stdout, stderr
        
    except Exception as e:
        return False, -1, "", str(e)

def check_complete_success(success, exit_code, stdout, stderr):
    # Check system-level success
    if not success or exit_code != 0:
        return False
        
    # Check for error indicators in output
    error_keywords = ["Error:", "Exception:", "Traceback", "failed"]
    for keyword in error_keywords:
        if keyword in stdout or keyword in stderr:
            return False
            
    # No system or application errors detected
    return True

# # Usage
# if check_complete_success(success, exit_code, stdout, stderr):
#     print("Command ran successfully with no exceptions")
# else:
#     print("Command encountered errors")

def run_shell_commands_parallel(commands, max_workers=None):
    """
    Run multiple shell commands in parallel and return their results.
    
    Args:
        commands: List of shell commands to run
        max_workers: Maximum number of worker threads (default: None, which uses ThreadPoolExecutor default)
        
    Returns:
        list: List of result tuples (success, exit_code, stdout, stderr) in the same order as input commands
    """
    # Use a list to store results in the same order as input commands
    results = [None] * len(commands)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all commands to the executor and track their indices
        future_to_index = {executor.submit(run_shell_command, cmd): i 
                          for i, cmd in enumerate(commands)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = (False, -1, "", str(e))
    
    return results

if __name__ == "__main__":
    commands = [
        "echo 'Hello World'",
        "ls -la",
        "sleep 2 && echo 'Done sleeping'"
    ]

    results = run_shell_commands_parallel(commands)

    breakpoint()
    for i, (success, exit_code, stdout, stderr) in enumerate(results):
        print(f"Command: {commands[i]}")
        print(f"Success: {success}, Exit code: {exit_code}")
        print(f"Output: {stdout}")
        if stderr:
            print(f"Error: {stderr}")
        print("-" * 40)