# Unravelling-Temporal-patterns   


Time series data refers to a collection of data points that are recorded, measured, or observed over a period of time. The accurate classification of these temporal sequences is crucial for informed decision-making in various domains such as weather data, finance, and healthcare (1)(5). This project explores two approaches for time series classification using Recurrence Quantification Analysis (RQA) and Recurrence Network measures with machine learning, and classification using recurrence plot images with Machine Learning.  

This project is a Python software with GUI for the following tasks:  
1) Generation of Time series data ( Time series Excel files  and also Recurrence plot images, Phase space plots, Reconstructed phase space plots, etc.)  
2) Perform RQA and Network Analysis on the data  
3) Train and evaluate Machine Learning Models to classify time series data (PERIODIC, CHAOTIC, NOISE, HYPERCHAOS) using two data formats: RQA (Recurrence Quantification Analysis) + Network Features  and Recurrence Plot Images.
5) Classify unseen data using the Trained Model.  
  
The tab 3 and tab 4 can be used for general machine-learning classification and prediction problems as well.   

I implemented CPU-based parallel processing to substantially reduce the computation time for Average Mutual Information and FNN calculations, which are essential for determining embedding delay and dimension. This enhancement is readily noticeable when comparing it to the previous version, RLV45.Use RLV50 for parallel processing features. All these features facilitate the computation of more than 1 million time series points. RLV50 also uses a CNN for Recurrence plot classification instead of SVM. 

Furthermore, I employed the PyRQA library (in RLV50), which has been documented as the most efficient RQA method compared to alternatives such as pyunicorn and manual approaches, except for approximate methods [2].  

CAUTION: THE PROGRAM MIGHT PUSH THE TEMPERATURE OF YOUR CPU TO VERY HIGH VALUES AND IT IS GOOD TO KEEP A CHECK OF THE CPU TEMPERATURES. I tested it on an externally watercooled laptop so it was not an issue for me.
   
It was developed as part of my project work under the guidance of Professor:  
Chandrakala Meena   
IISER TVM.    


 


## Table of Contents
- [Methodology](#Methodology)
  - [Systems Considered](#Systems-Considered)
  - [Transient-Removal](#Transient-Removal)
  - [Delay Embedding](#Delay-Embedding)
    - [Delay](#Delay)
    - [Dimension](#Dimension)
  - [RQA-Network](#RQA-Network)
    - [Recurrence-Threshold](#Recurrence-Threshold)
       -[Using Standard deviation of series.](#Recurrence-Threshold)
       -[fixed percentage recurrence rate](#Fixed-percentage-recurrence-rate)
    - [RQA-Network_Measures](#RQA-Network_Measures)
  - [Machine-Learning-Algorithms](#Machine-Learning-Algorithms)
    - [RQA-Network-Based-Classification](#RQA-Network-Based-Classification)
    - [Recurrenceplot-Based-Classification](#Recurrenceplot-Based-Classification)
- [Installation](#Installation)
  - [Using-Python](#Using-Python)
  - [Using-Anaconda](#Using-Anaconda)
- [UI-Guide and WorkFLow](#UI-Guide)
  - [To Train and Evaluate ML Models](#Train-and-Evaluate-ML-Models)
    - [Tab-1](#Tab-1)
    - [Tab-2](#Tab-2)
    - [Sorting of Data for RQA-Network method](#Sorting-of-Data-for-RQA-Network-method)
    - [Sorting of Data for Recurrence Plot method](#Sorting-of-Data-for-Recurrenceplot-method)
  
    - [Tab-3](#Tab-3)
    - [Tab-4](#Tab-4)
  - [Make Predictions](#Make-Predictions)
     - [Sorting of Data for RQA-Network method](#Sorting-of-Data-for-RQA-Network-method_P)
     - [Sorting of Data for Recurrence Plot method](#Sorting-of-Data-for-Recurrenceplot-method-P)
  - [Good practices and Known-Issues](#Good-practices-and-Known-Issues)
- [Publications](#Publications)
- [References](#References)
- [Issues](#issues)
- [License](#licensing)

# Methodology
## Systems-Considered
Systems Available for generation of time series data:    
All time Series Data is normalized.  
1)Lorenz  [1]  

The Lorenz system is a set of three differential equations described as:  
$\frac{dx}{dt}=sigma*(y-x)$  
$\frac{dy}{dt}=x*(rho-z)-y$  
$\frac{dz}{dt}=x*(y)-(beta*z)$  
sigma = 10  
beta = 8/3   
rho is the varied parameter.   

2)Rossler    [2]  
$\frac{dx}{dt} = -y - z$  
$\frac{dy}{dt} = x + a \cdot y)$  
$\frac{dz}{dt} = b + z \cdot (x - c)$  
a = 0.2  
b = 0.2  
c is the varied paramter.  

3)Duffing    [3]  
$\frac{dx}{dt} = v$  
$\frac{dv}{dt} = x - x^3 - \delta \cdot v + a \cdot \sin(\omega t)$  
delta = 0.5   
omega= 1  
a is the varied parameter.  

4)Noise    
For noise, I have added Gaussian random noise to a sinewave with 0 amplitude.  

5)Empty plots    
Empty plots were employed to verify the distinction between noise recurrence plots and those with longer noise time series files. With extended noise time series data, it becomes challenging to discern the images, as they often appear nearly empty. This is particularly evident when using an embedding dimension of 4 for the noise and a threshold set at 40 percent of the standard deviation of signal x.  

6)Chen system ( for hyperchaos)    [4]  
$\frac{dx}{dt}=a\cdot(y-x)+e\cdot y\cdot z$  
$\frac{dy}{dt}=c\cdot x-d\cdot x\cdot z+y+u$  
$\frac{dz}{dt}=x\cdot y-b\cdot z$  
$\frac{du}{dt}=-k\cdot y$    
a = 35  
b = 4.9  
c = 25  
d = 5  
e = 35  
k is the varied parameter

## Transient-Removal  
Phase space with transient behaviour-  
  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/323c4b5a-76ab-4892-8f3a-0ffc672eeed0)
  
  
Phase space with transient behaviour removed-   
  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/36d16ef0-f7e7-4627-8f89-3d820e94d014)
  


For Transient behaviour removal, I have used the below two methods :  
1)Use the final condition of an iteration as the initial condition of the next parameter iteration.  
2)Removed 50 percentage of initial time-series data points.  

## Delay-Embedding  
Method used for estimation of embedding delay and embedding dimension:   
  
### Delay   
For Optimal embedding delay one can use two of the options below in the user interface :  
1)Average Mutual information [5].  




2)Autocorrelation method   




### Dimension   
For Optimal embedding delay the below method is used :  
1)False Nearest Neighbours Method- [6]   

Perfromance stats for this is not loggged yet but a considerable difference is also seen when using multi core cpu.  
 
## RQA-Network  

### Recurrence-Threshold
The recurrence threshold can be found in two different ways:  
### Using-Standard-deviation-of-series
The recurrence threshold is taken as a percentage of the standard deviation of the signal. The user can set the percentage as an input [7].   
### Fixed-percentage-recurrence-rate
The bisection algorithm is used to find the right threshold value using pyRQA. The algorithm works as follows:  
It starts with an initial guess for the threshold, threshold, which is set between 0.0 and the maximum value of the time series xL.  
The code defines a function  that computes the actual recurrence rate using the RQA analysis with the current threshold.  
The bisection loop iterates until the  difference between the actual recurrence rate and the desired recurrence rate (10%) is less than a predefined tolerance, which is set to 1e-4. This can be set to bigger values for better performance.  
If the solution does not converge there will be an error message. This method is slower than the one which uses the standard deviation of signal.

### RQA-Network_Measures  
The recurrence matrix is used as the adjacency matrix to construct nodes with self-loops removed using simple graph (igraph).  

The following RQA and Network measures are calculated :   

Recurrence Rate  
Determinism  
Laminarity	 
Trapping Time   
LMAX	   
Degree Distribution	   
degree centrality	   
average path length     
Clustering Coefficient     
Diagonal Line Entropy   
Network Density     
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/29adcfe5-c985-4ffc-b5df-2ebbc5c0cefc)  





## Machine-Learning-Algorithms  
### RQA-Network-Based-Classification  
CLassifiers Used for RQA and Network Numerical Data based Training and Predictions:    
Decision Tree  
Random Forest  
KNN  
SVM (linear and RBF kernel)  
Note : Feature Importance for Decision Tree and Random Forest are also plotted and saved.   

### Recurrenceplot-Based-Classification  
CLassifiers Used for Recurrence Plot Training and Predictions : CNN
















## Installation  
Instructions on how to install the project.  



## Using-Python    
1) Download the Python main (RLV45) file.    

2) Install Python from - https://www.python.org/downloads/   
    Tick the button saying add Python to path in the installation window for Python installation.  

    If Python is already installed and was not added to path simply uninstall and perform the above steps.   



3) Open command prompt on Windows:   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/3d5f24b6-00f9-4425-807f-263ece9e9f1a)   

Please copy the commands below (using top right icon below), paste (right click ) them into Windows  command prompt window, and run (Enter) them to install the necessary dependencies :  

```bash
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install python-igraph
pip install igraph
pip install PyRQA
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install python-igraph
pip install openpyxl
pip install pycairo
pip install numpy
pip install scipy
pip install pillow
pip install joblib
pip install tensorflow
pip install kneed
```
4) Once the libraries are installed double click on the downloaded python file (RLV45) to run the program.  

Install any other libraries if the program gives any  missing error in the console window.  


## Using-Anaconda  
Install Anaconda distribution from- https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Windows-x86_64.exe   

Then open  the Anaconda prompt (search in Windows search bar) > Install the below dependencies using the commands  
```bash
pip install igraph
pip install PyRQA
```
Open Anaconda Navigator > Open the main python file >run  
OR
Navigate to the folder containing the main  python  and run it using :  
```bash
cd path/to/your/project/directory
python main.py
```


Executable : Still in progress  


## UI-Guide  

### Train-and-Evaluate-ML-Models  

#### Tab-1
Tab1: Time Series Generation
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/f47a42c1-5ff9-4ad1-b883-3180d97e45cf)

  

Within this section, you can generate pure time series data that belong to specific classes, such as periodic or chaotic. Here's how it works:  

Use the dropdown menu to select the system for generating time series data and specify the parameter you want to vary:  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/04939fc9-ab6a-418f-81d9-36d0d2e63955)

 
One can inject noise into the time series using a noise radio button or generate noise-only data from the systems available. The standard deviation of noise can be custom picked and visualized :  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/313c8174-82a6-4e14-8859-9c7760f546c4)


 


Input the parameter range by providing lower and upper values in the provided entry boxes. This customization helps in obtaining time series data with specific class characteristics.  
Set the threshold for the standard deviation of the signal as a percentage (%). This threshold is crucial for recurrence detection.  
The Status tab keeps you informed about the current program status.  
The Console provides real-time updates and other program outputs, excluding plots.  
Various plots, including Time Series, Phase Space, Recurrence Plots, Reconstructed Plots, and Mutual Information Function vs. Delay Plots, are automatically saved upon program completion. You will be prompted to choose a location for saving these plots.   
You can easily generate the folder structure using the Create folders/Subfolders button.   
Here is how the folders will be generated using the Create Folders/Subfolders button:    
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/8579adab-77d1-4989-bb24-27e555b1f2d1)
  
Here is how the files will be saved:    
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/cceab99c-e5be-4e98-9c03-4ba21748333a)


 
Processing flow:  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/94eb9c9a-03a3-4568-ad71-822e7f312efe)  





#### Tab-2  
Tab2: RQA and Network Analysis   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/39c7f283-8d6b-4104-b436-44252720e60a)

   


This tab performs Recurrence Quantification Analysis (RQA) and network analysis. Here's how it functions:  
The recurrence plot radio button can be used to generate a recurrence plot in this tab as well as it is needed for unseen data. Make sure to pick the same method used for finding embedding delay here as well in the drop-down menu.  
Click the RQA measures and Network Measures button  and navigate to the folder where the time series data was generated and select it.  
Upon completion, the program will create a new Excel file in the same location as shown below:    
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/34efd3fa-0b35-4865-8f87-49bc32287694)
  

Processing Flow :  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/bd9fe671-fa26-41dd-ab21-d0530005c691)  



#### Sorting-of-Data-for-RQA-Network-method   
In each class folder inside each system we now have the RQA and network measures:  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/64bab96c-69a0-436a-82f4-6c9c4b633ff1)
   

Add a class column to the end based on the corresponding class the file is from and remove the parameter (rho,c etc.) column, Network density, Degree Distribution, and Degree Centrality as shown below :   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/5a5583d2-2d6c-4d84-b171-ef65a145a21f)
  


Repeat for all RQA and Network Excel files and  combine all the RQA and Network file data into a single Excel file with a single header row like:   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/edf981cd-cd54-460c-be7a-55103447d5af)
   

Now we have the features ready for  training the Machine Learning algorithms in tab3.  

#### Sorting-of-Data-for-Recurrenceplot-method  
Make a folder inside a parent folder with the Class names as shown below:  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/8275aa5d-0884-4021-9c8d-f4587d6320cf)
  


Available Class names are :   
    PERIODIC   
    CHAOTIC  
    NOISE   
    EMPTY  
    HYPERCHAOS  
    NEW1   
    NEW2  
    NEW3  
    NEW4  
    NEW5  
  
Classes named NEW are used in case a new CLASS is needed in future for some classification task. Copy and paste the needed class names exactly as folder names inside the parent folder so that the program can identify them.  
Now add the recurrence plots belonging to these classes from the plot folder for each system to these folders accordingly.  
Now we can use these for Training in tab 4.  




#### 3 Tab-3  
Tab3: Machine Learning - RQA and Network Measures   
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/042b9dc2-018c-4b36-9e5f-bf7c2bd34db8)
  

In this tab, you can perform machine learning tasks related to RQA and network measures:  

The Train/Test button trains and evaluates various machine learning models, such as Random Forest, Decision Tree, SVM (Linear and RBF), and KNN. When clicked it will ask the user to select the sorted Excel file from before.  
It also saves the trained models using Joblib in the same location as the sorted Excel file.  

#### Tab-4  
Tab4: Image Classifier SVM  
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/b48f7b31-13d6-4dd8-898e-774c97a2bacc)
 



This tab is specifically designed for image classification using Support Vector Machines (RLV45) or CNN (RLV50). Here's how it operates   



The first Train/Test button trains and evaluates different machine learning models using Recurrence Plot images, including Random Forest, Decision Tree, SVM (Linear and RBF), and KNN.    
It also saves the trained models using Joblib.   
The second Run Image Predictor button is used for making predictions on new Recurrence Plot images.   



## Make-Predictions  
### Sorting-of-Data-for-RQA-Network-method_P  
Suppose we have unseen time series data.  
1) Save it as an Excel file in the same way the time series file was saved in the tab 1 operation.  
2) Rename the file using- time_series_a_1    
3) if there are multiple then rename like- time_series_a_1 , time_series_a_2 etc.  
4) Now go to tab 2 and the output Excel file with RQA and network measures will be saved in the same folder as our unseen time series file.  
5) Sort this the same way as before but there is no need to add a CLASS column as this is what we get from predictions.  
6) Proceed to tab3 and use the trained model and This new Excel file to make predictions. The result will also be saved as an Excel file with an added column with the predictions.  

Sorting-of-Data-for-Recurrenceplot-method-P    
1) Click on the Generate Recurrence plot radio button in tab 2 and do the same steps 1-4 from  Sorting-of-Data-for-RQA-Network-method.  
2) Now take all the recurrence plots saved in the folder same as the unseen time-series data folder and move them to a new folder.  
3) Go to tab 4 and use the model saved for recurrence plot-based classification and also the recurrence plot images and make predictions  
    
If there are multiple Recurrence Plot images for prediction, you can utilize the slider within the same window which will pop up once the predictions are made: 
![image](https://github.com/am0032/Unravelling-Temporal-Patterns/assets/123314532/b99644f9-50ae-4813-b7fc-8258b73288f1)


Predictions are displayed as text overlays in the same window and are also saved in a new folder at the same location.  
   



## Good-practices-and-Known-Issues  


In tab 1 restart the UI after it's used to generate Excel files in each class.  

Before closing the UI go to the respective folder where Excel files are generated and make sure all files are there. It might take some time to write the Excel files. Once checked  the program can be closed.  

Make sure to balance the number of data files in each class to roughly the same number  

Error that shows up in the console when you press the generate time series on tab 1. This can be ignored.  

When saving the plots have a significant amount of border which if  removed should give better results for recurrence plot based classification.

# issues  
Tab 4 based on the recurrence plot is memory intensive and might crash on low RAM. Currently working on this to find a solution.  


# References  

(1)References Wikimedia Foundation. (2023d, August 13). Rössler attractor. Wikipedia. https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor   

(2) Marwan, N., Kraemer, K.H. Trends in recurrence analysis of dynamical systems. Eur. Phys. J. Spec. Top. 232, 5–27 (2023). https://doi.org/10.1140/epjs/s11734-022-00739-8

(3)References Wikimedia Foundation. (2023b, July 23). Duffing equation. Wikipedia. https://en.wikipedia.org/wiki/Duffing_equation   

(4)Chen, Z., Yang, Y., Qi, G., &amp; Yuan, Z. (2007). A novel hyperchaos system only with one equilibrium. Physics Letters A, 360(6), 696–701. https://doi.org/10.1016/j.physleta.2006.08.085   

(5)References L., W. Jr. C., & Marwan, N. (2015). Recurrence quantification analysis theory and best practices (pp. 8). Springer.  

(6)Kennel, M. B., Brown, R., &amp; Abarbanel, H. D. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical Review A, 45(6), 3403–3411. https://doi.org/10.1103/physreva.45.3403   

(7)References Wikimedia Foundation. (2023f, August 18). Lorenz System. Wikipedia. https://en.wikipedia.org/wiki/Lorenz_system 

