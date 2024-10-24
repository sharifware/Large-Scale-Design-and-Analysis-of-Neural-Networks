# 10/10/2024
-	Went over current code to find next steps
-	Researched ways to remove jupyter outputs from git (this still needs to be implemented)
-	Modified existing code to accept dynamic config file 
o	https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks/commits/Config-File
-	Added environment variables implementation
o	https://github.com/sharifware/Large-Scale-Design-and-Analysis-of-Neural-Networks/commits/environment-variables

# 10/13/2024
- Wrote script to generate data for the simple regression function: $f(a, b) = \frac{1}{5} a^2 - \frac{1}{10} b^3$
- saves as csv
- updated inputNetwork.ipynb to:
-  load data as tensor
- split data into train and test
- dynamically instantiate one NN as test
- Added new architecture file, simpleRegArchitecture.py
- modified code to instantiate n number of networks specified in the config file
- added function to train a network and print how the loss is progressing across epochs
- This function is iteratively called for each instantiated network

# 10/20/2024
-	Planned binning of weights
-	Reconfigured github repository to have develop and main branches
-	Modified pipeline to save trained networks as a state dict
-	Loaded saved networks into new file, binningWeights.ipynb
-	A question I have for John: Should the min and max values for the weights be dynamic based on the data where the min is the smallest data point and the max is the largest or should the min and max be a set number specified by the user like 0 and 1? Currently I’ll implement like the min and max are dynamic
-	Wrote functions to generate the bins based on the weights in the loaded network and then populate the counts of weights in those bins.
-	Made simple histogram with this data

# 10/22/2024
- Realized in class today during our meeting with John that I was really misunderstanding how to bin the weights, no wonder it was suspiciously simple
- I was simply aggregating all the weights for all the networks then creating bins and histograms
- I'll have to instead create bins for each weight in the architecture of the networks and populate those bins with data aggregated from all the networks.
- This was the plan the whole time and I knew that, I just wasn't thinking it through all the way earlier.
- This will be more complex
- I've written loops to iterate through and create an empty matrix of zeroes, however I think there's an issue with the shape of it
- I was struggling for a while with creating empty bins of the right shape for all the layers
- The approach I've taken is to gnerate the data for each layer separately then add them to an array

# 10/23/2024
- After a lot of thinking I figured out how to create empty matrices of the correct shape and how to save all the weights in the shape I need
- I've finished the iteration through populating the bins, it's gotten pretty complex but I think it's working
- The code could definitely do for some fresh eyes looking at it becuause it did get pretty complex
- implemented initial histogram generation for each weight but it's not really working yet, need to look at it more in a future sprint, as planned.
- Realized we never actually started the Jira sprint, in the future we should definitely need to be more on top of Jira
- Made branch for the notebooks
