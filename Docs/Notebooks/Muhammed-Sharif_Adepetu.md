
# 10/20/2024  
- Began designing functions for visually assessing the loss of various neural networks during training.  
- Created a simple loss graph using data from a single neural network as a proof of concept to test the function's basic functionality.  

# 10/22/2024  
- Realized I had written the graphing function in the wrong branch, so I created a new branch specifically to test the loss function graphs.  
- Started planning how to visualize the loss for all networks in the dataset. Decided that each network's loss should be represented by a separate line on the same graph.  
- Planned to store all networks in an array and track their respective loss values for easier plotting.  

# 10/23/2024  
- Considered what Jhon might view as the success criteria for the loss visualization. For now, decided to track the loss over the first 100 iterations to gather initial data.   
- Spent time retroactively documenting past work for the last few weeks in the engineering notebook branch, as I had neglected to take detailed notes earlier.  

# 11/04/2024  
- Jhon clarified that the visualization should allow users to input the number of epochs they want to see the loss change for.

# 11/07/2024  
- Modified the loss function criterion to use Mean Squared Error (MSE) for evaluating the loss during training.  

# 11/10/2024  
- Reviewed the software design documen. Made  notes about updates needed.  
- Updated Jira to reflect recent changes and progress on tasks.  

# 11/13/2024  
- Created a new branch off Daniel's branch to integrate the visualization function into his class body.  
- Fixed an indexing issue with how epochs were being handled, which was causing incorrect visualizations in earlier iterations.  

# 11/16/2024  
- Continued refining the software design document, focusing on sections that required further elaboration based on recent discussions and testing.  
- Began testing input features for dynamically selecting the number of epochs for visualization. Validated initial test cases for usability and correctness.  

# 11/19/2024  
- Re-merged my provided code with Daniel's.  
- Added detailed notes to the methodology section of the final report to outline the visualization function's purpose, implementation, and expected outcomes.

# 11/21/2024  
- Made additional revisions to the software design document.  
- Updated Jira to reflect the conclusion of the sprint. 

# 11/26/2024  
- Worked on the Software Requirements Specification (SRS) document, incorporating the TAs' comments about adding more detail to the functional requirements and user scenarios.  

# 11/28/2024  
- Revised the project poster based on feedback from TA. Made changes to the visual design and ensured all graphs and diagrams were properly labeled.  
- Updated the PowerPoint to include additional testing data and a clearer breakdown of the project's contributions.  

# 12/2/2024  
- Worked on final project presenation. added design considerations and subsystem design slides.

# 01/09/2025 
-Reviewed weight binning notebook, making modifications to the program and dividing different parts into methods that the user will be able to utilize. 
-Begun transforming other notebooks into scripts.

# 01/23/2025  
- Watched tutorial on package creation
- Set up directory to make package. 

# 01/28/2025
- Finalized scripts from notebooks. begun writing a test file that will test the main functionality of both network generation and weight binning modules.

# 02/03/2025
- Wrote multiple sections of SRS 
  
# 02/04/2025  
- Re-Structured Project files to comply with standard practices
- Updated setup.py file for package
- Wrote LICENCE.txt file

# 02/05/2025
- Included tested network analyzer file and verified the output.
- Cleaned directory. changed initialization method to include architecture type and access the networks that have been created from NetworkANalyser
- Edit, src, buld and dist files to include propper files

# 02/10/2025  
- Added proper path information to class body and tested plotting

# 02/11/2025  
- Verified Network generation and histogram saving modules worked within package.
- investigated why the network generation performance was poor under certain conditions

# 02/18/2025
- Corrected save directory location.
- uploaded pacakge to TestPYPI

# 02/20/2025
- Fixed bug that occured when user tried installing pacakge from piPY
- Began Working on testplan
- 
# 03/05/2025  
- Fixed Wheel and Distribution Files for the Package:
- Resolved issues with the setup.py and pyproject.toml configuration.
- Successfully built .whl and .tar.gz distribution files using build module.
 
# 03/25/2025  
- Uploaded the package using twine to the TestPyPI repository.
- Confirmed successful upload and visibility on TestPyPI.

# 04/03/2025
- Created an isolated conda Jupyter environment and Installed the package directly from TestPyPI.
- Executed test cases to validate core functionality and integration.
- Confirmed consistent behavior with development version and no runtime errors

# 04/10/2025
- Documented performed test cases

# 04/22/2025
-  performed and documented rest of performed test cases

# 04/24/2025
- deployed officical package to PyPI.


 
