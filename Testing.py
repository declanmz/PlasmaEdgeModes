import time
from tqdm import tqdm

# Define the dimensions for the nested loops
outer_iterations = 10
inner_iterations = 5

# Initialize tqdm for the outer loop
outer_pbar = tqdm(total=outer_iterations, desc="Outer Loop", position=0)

outer_counter = 0
while outer_counter < outer_iterations:
    # Initialize tqdm for the inner loop
    inner_pbar = tqdm(total=inner_iterations, desc="Inner Loop", position=1, leave=False)
    
    inner_counter = 0
    while inner_counter < inner_iterations:
        # Simulate some work with a tenth of a second delay
        time.sleep(0.1)
        
        # Update the inner counter
        inner_counter += 1
        
        # Update the inner progress bar
        inner_pbar.update(1)
    
    # Close the inner progress bar
    inner_pbar.close()
    
    # Update the outer counter
    outer_counter += 1
    
    # Update the outer progress bar
    outer_pbar.update(1)

# Close the outer progress bar
outer_pbar.close()

print("Process complete!")