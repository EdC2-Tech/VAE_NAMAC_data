# VAE_NAMAC_data

## Description
This program generates training data using a variational autoencoder to train machine learning models. The data generated is based off a GOTHIC simulation of the Experimental Breeder Reactor II. Experimental validation of the GOTHIC simulation is available at:
  J.W. Lane, J.M Link, J.M. King, T.L. George, S.W. Claybrook, "Benchmark of GOTHIC to EBR-II SHRT-17 and SHRT 45R Tests," Nuclear Technology, https://doi.org/10.1080/00295450.2019.1698896
  
Three types of data are generated; total core flow rate [kg/s], upper plenum temperature [C], and fuel centerline temperature [C].


## How to generate data
### Step 1:
It is recommended to use Anaconda to generate the necessary dependencies to run the file. The environment information to generate the necessary dependencies is provided in the "environment.yml" file.
Ensure that you setup the environment using the .yml file so that the program can work properly.

### Step 2:
Open "DataGen.py" and run as is.

### Step 3:
The generated data is located in the folder "Generated_Data". Details about each column is provided in the file "Column Information.txt"

### Step 4 (Optional):
Use the data to construct your own machine learning model. The input features are the total core flow rate and the upper plenum temperature. The target is the fuel centerline temperature. Time is not a feature and is only used for plotting.

## Repository Contents
