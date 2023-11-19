from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import tkinter.font as font
from tkinter import filedialog
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scipy.signal import argrelextrema
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeClassifier
import os
import time
import igraph as ig
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import entropy 
from sklearn.preprocessing import MinMaxScaler
from tkinter import OptionMenu, StringVar
from PIL import  ImageTk
from tkinter import  messagebox
from sklearn.neighbors import NearestNeighbors
import joblib
from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import  GridSearchCV
stddvofsignal=0
fixed_RR=0
rr_stddev=0
case = 0
A = 0
B = 0
var = 0
n= 45 #random state


# UI
root = tk.Tk()
noise_value=0


import tkinter as tk
import sys


def write_to_text_widget(text_widget, text):
    text_widget.config(state=tk.NORMAL)  # Enable editing
    text_widget.insert(tk.END, text)     # Insert the text
    text_widget.config(state=tk.DISABLED)  # Disable editing to make it read-only
    text_widget.see(tk.END)  # Scroll to the end of the Text widget

# Redirect stdout and stderr to the Text widget
sys.stdout.write = lambda text: write_to_text_widget(output_text, text)
sys.stderr.write = lambda text: write_to_text_widget(output_text, text)



# Create a MinMaxScaler object
scaler = MinMaxScaler()

def update_status1(status):
    status_label.config(text=status)

notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, columnspan=1, padx=0, pady=0, sticky="nsew")

# Create a custom style for the notebook to set the background color
notebook_style = ttk.Style()
notebook_style.theme_use('default')
notebook_style.configure('TNotebook.Tab', background='#419292', foreground='black', font=('Helvetica', 10, 'bold'))

# tab for the first program
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Time Series Generation")

#tab for the second program
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="RQA and Network analysis/Feature Generation")

#tab for the 3rd program
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text=" Machine Learning Numerical data Train and Prediction")

#tab for the 4th program
tab4 = ttk.Frame(notebook)
notebook.add(tab4, text="Machine Learning Image data Train and Prediction")

#tab for the 5th program
tab5 = ttk.Frame(notebook)
notebook.add(tab5, text="About")


#background color of the widgets within each frame
tab1.configure(style='Custom.TFrame')
tab2.configure(style='Custom.TFrame')
tab3.configure(style='Custom.TFrame')
tab4.configure(style='Custom.TFrame')
tab5.configure(style='Custom.TFrame')


# Create a custom style for the frames to set the background color
frame_style = ttk.Style()
frame_style.theme_use('default')  # You can use any available theme
frame_style.configure('Custom.TFrame', background='#1e1f26')


# Configure row and column weights to make the notebook and its contents expand
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)


# the label to explain how the program works 
explanation_label = tk.Label(tab1, text="This tab varies one parameter and solves a dynamical system.")
explanation_label.grid(row=0, column=0, columnspan=2, padx=5, pady=10, sticky="nsew")
explanation_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a list of options for the drop-down menu (content for Program 1)
options1 = ["Lorenz-rho varied, sigma and beta constant","Rossler-c varied, a and b constant","Duffing-a varied ", "Noise","Empty","Chen (hyperchaos)"]

# Create a StringVar to store the selected option 
selected_option1 = tk.StringVar()

# Set the initial value of the selected option to the first item in the options list 
selected_option1.set(options1[0])

# Create the drop-down menu 
drop_down_menu = tk.OptionMenu(tab1, selected_option1, *options1)
drop_down_menu.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
drop_down_menu.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a label and entry widget for variable A 
a_label = tk.Label(tab1, text="Lower Limit of varied parameter:")
a_label.grid(row=5, column=0, padx=5, pady=5, sticky="nsew")
a_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
a_entry = tk.Entry(tab1)
a_entry.grid(row=5, column=1, padx=5, pady=5, sticky="nsew")
a_entry.config(bg="#1e1f26", fg="#8c9da9")

# Create a label and entry widget for variable B 
b_label = tk.Label(tab1, text="Upper Limit of varied parameter:")
b_label.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
b_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
b_entry = tk.Entry(tab1)
b_entry.grid(row=6, column=1, padx=5, pady=5, sticky="nsew")
b_entry.config(bg="#1e1f26", fg="#8c9da9")




def show_entry_tab1(*args):
    if rr_stddev_tab1.get() == "Fixed Recurrance rate":
        fixed_RR_label_tab1.grid(row=9, column=0, padx=1, pady=5, sticky="nsew")
        fixed_RR_entry_tab1.grid(row=9, column=1, padx=5, pady=3, sticky="nsew")

        stddvofsignal_label_tab1.grid_forget()  
        stddvofsignal_entry_tab1.grid_forget()

    elif rr_stddev_tab1.get() == "Percentage of Standard deviation of signal to use as threshold":
        stddvofsignal_label_tab1.grid(row=9, column=0, padx=1, pady=5, sticky="nsew")
        stddvofsignal_entry_tab1.grid(row=9, column=1, padx=5, pady=3, sticky="nsew")
        fixed_RR_label_tab1.grid_forget() 
        fixed_RR_entry_tab1.grid_forget()
    else:
        stddvofsignal_label_tab1.grid_forget()
        stddvofsignal_entry_tab1.grid_forget()
        fixed_RR_label_tab1.grid_forget()
        fixed_RR_entry_tab1.grid_forget()

def show_entry_tab2(*args):
    if rr_stddev_tab2.get() == "Fixed Recurrance rate":
        fixed_RR_label_tab2.grid(row=9, column=0, padx=1, pady=5, sticky="nsew")
        fixed_RR_entry_tab2.grid(row=9, column=1, padx=5, pady=3, sticky="nsew")

        stddvofsignal_label_tab2.grid_forget()  
        stddvofsignal_entry_tab2.grid_forget()

    elif rr_stddev_tab2.get() == "Percentage of Standard deviation of signal to use as threshold":
        stddvofsignal_label_tab2.grid(row=9, column=0, padx=1, pady=5, sticky="nsew")
        stddvofsignal_entry_tab2.grid(row=9, column=1, padx=5, pady=3, sticky="nsew")
        fixed_RR_label_tab2.grid_forget()  
        fixed_RR_entry_tab2.grid_forget()
    else:
        stddvofsignal_label_tab2.grid_forget()
        stddvofsignal_entry_tab2.grid_forget()
        fixed_RR_label_tab2.grid_forget()
        fixed_RR_entry_tab2.grid_forget()


# Create a StringVar to store the selected option from the dropdown menu for tab1
rr_stddev_tab1 = tk.StringVar()

# Set the initial default value for the dropdown menu in tab1
rr_stddev_tab1.set("Percentage of Standard deviation of signal to use as threshold")

# Create a StringVar to store the selected option from the dropdown menu for tab2
rr_stddev_tab2 = tk.StringVar()

# Set the initial default value for the dropdown menu in tab2
rr_stddev_tab2.set("Percentage of Standard deviation of signal to use as threshold")

# Create a label
threshold_label_tab1 = tk.Label(tab1, text="Method to set Recurrence Threshold")
threshold_label_tab1.grid(row=8, column=0, padx=5, pady=5, sticky="nsew")
threshold_label_tab1.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

threshold_label_tab2 = tk.Label(tab2, text="Method to set Recurrence Threshold")
threshold_label_tab2.grid(row=8, column=0, padx=5, pady=5, sticky="nsew")
threshold_label_tab2.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

threshold_options = ["Fixed Recurrance rate", "Percentage of Standard deviation of signal to use as threshold"]

# Create a dropdown menu for tab1
dropdown_tab1 = tk.OptionMenu(tab1, rr_stddev_tab1, *threshold_options)
dropdown_tab1.grid(row=8, column=1, padx=5, pady=5, sticky="e")
dropdown_tab1.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a dropdown menu for tab2
dropdown_tab2 = tk.OptionMenu(tab2, rr_stddev_tab2, *threshold_options)
dropdown_tab2.grid(row=8, column=1, padx=5, pady=5, sticky="e")
dropdown_tab2.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a label and entry widget for the standard deviation of the signal in tab1 (initially hidden)
stddvofsignal_label_tab1 = tk.Label(tab1, text="% of Standard deviation of signal to use as threshold:")
stddvofsignal_label_tab1.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
stddvofsignal_entry_tab1 = tk.Entry(tab1, width=3)  # Set the width as desired
stddvofsignal_entry_tab1.config(bg="#1e1f26", fg="#8c9da9")
stddvofsignal_entry_tab1.insert(0, "0.5")

# Create a label and entry widget for the standard deviation of the signal in tab2 (initially hidden)
stddvofsignal_label_tab2 = tk.Label(tab2, text="% of Standard deviation of signal to use as threshold:")
stddvofsignal_label_tab2.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
stddvofsignal_entry_tab2 = tk.Entry(tab2, width=3)  # Set the width as desired
stddvofsignal_entry_tab2.config(bg="#1e1f26", fg="#8c9da9")
stddvofsignal_entry_tab2.insert(0, "0.5")

# Create a label and entry widget for the fixed recurrance rate in tab1 (initially hidden)
fixed_RR_label_tab1 = tk.Label(tab1, text="Fixed Recurrance rate:")
fixed_RR_label_tab1.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
fixed_RR_entry_tab1 = tk.Entry(tab1, width=3)  # Set the width as desired
fixed_RR_entry_tab1.config(bg="#1e1f26", fg="#8c9da9")
fixed_RR_entry_tab1.insert(0, "0.10")

# Create a label and entry widget for the fixed recurrance rate in tab2 (initially hidden)
fixed_RR_label_tab2 = tk.Label(tab2, text="Fixed Recurrance rate:")
fixed_RR_label_tab2.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
fixed_RR_entry_tab2 = tk.Entry(tab2, width=3)  # Set the width as desired
fixed_RR_entry_tab2.config(bg="#1e1f26", fg="#8c9da9")
fixed_RR_entry_tab2.insert(0, "0.10")

# Call show_entry_tab1 and show_entry_tab2 to display the default option when the program starts for both tabs
show_entry_tab1()
show_entry_tab2()

# Trace changes to the dropdown_var for both tabs and show/hide the label and entry widgets accordingly
rr_stddev_tab1.trace("w", show_entry_tab1)
rr_stddev_tab2.trace("w", show_entry_tab2)





# Create a label for variable to store number of excel files
noofplots_label = tk.Label(tab1, text="Number of Excel files to Generate:")
noofplots_label.grid(row=7, column=3, padx=50, pady=0, sticky="w")  
noofplots_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))

# Create an entry widget to entry number of excel files
noofplots_entry = tk.Entry(tab1, width=3)  # Set the width as desired
noofplots_entry.grid(row=7, column=4, padx=0, pady=0, sticky="w") 
noofplots_entry.config(bg="#1e1f26", fg="#8c9da9")
noofplots_entry.insert(0,"25")



selected_option2 = tk.StringVar()




# Create a variable to hold the selected value (content for Program 1)
noise_value = tk.IntVar()

style = ttk.Style()
style.configure("Dark.TRadiobutton",
                background="#1e1f26",
                foreground="white")


#radio button for noise 
noise_radio_button1 = ttk.Radiobutton(tab1, text="Noise", variable=noise_value, value=1)
noise_radio_button1.grid(row=2, column=0, padx=10, pady=5)
noise_radio_button1.configure(style="Dark.TRadiobutton")  # Add this line to set the background color

#the radio button for no noise 
noise_radio_button2 = ttk.Radiobutton(tab1, text="No Noise", variable=noise_value, value=0)
noise_radio_button2.grid(row=2, column=1, padx=10, pady=5)
noise_radio_button2.configure(style="Dark.TRadiobutton")  # Add this line to set the background color





# Create a variable to hold the selected value (content for Program 1)
recurrenceplot = tk.IntVar()

style = ttk.Style()
style.configure("Dark.TRadiobutton",
                background="#1e1f26",
                foreground="white")


#radio button for rp
recurrenceplotbutton1 = ttk.Radiobutton(tab2, text="Recurrence plot", variable=recurrenceplot, value=1)
recurrenceplotbutton1.grid(row=2, column=0, padx=10, pady=5)
recurrenceplotbutton1.configure(style="Dark.TRadiobutton")  # Add this line to set the background color

#the radio button for no rp
recurrenceplotbutton2 = ttk.Radiobutton(tab2, text="No Recurrence plot", variable=recurrenceplot, value=0)
recurrenceplotbutton2.grid(row=3, column=0, padx=10, pady=5)
recurrenceplotbutton2.configure(style="Dark.TRadiobutton")  # Add this line to set the background color



# Add  options
options4 = ["Mutual-Information", "Autocorrelation"]  

# Create a Var to store the selected option
selected_option4 = StringVar()

# Set the initial value of the selected option 
selected_option4.set(options4[0])

# Create the dropdown menu for the new options
drop_down_menu4 = OptionMenu(tab1, selected_option4, *options4)
drop_down_menu4.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")
drop_down_menu4.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))
# Create the dropdown menu for the new options
drop_down_menu4 = OptionMenu(tab2, selected_option4, *options4)
drop_down_menu4.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")
drop_down_menu4.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))




# Add the new options 
options2 = ["SOLVER:DOP853", "SOLVER:RK45"]  # Add your desired options

# Create a Var to store the selected option
selected_option2 = StringVar()

# Set the initial value of the selected option
selected_option2.set(options2[0])

# Create the dropdown menu for the new options
drop_down_menu2 = OptionMenu(tab1, selected_option2, *options2)
drop_down_menu2.grid(row=4, column=3, padx=50, pady=2, sticky="nsew",)
drop_down_menu2.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"),width=5)



# Create  label 
delay_label = tk.Label(tab1, text="Method to Find Embedding Delay:")
delay_label.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
delay_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))





import tkinter as tk
from tkinter import Scale


def slider_Timelength_changed(event):
    valuelength = slider_var_length.get()
    


# variable to be controlled by the slider
slider_var_length = tk.DoubleVar()

# slider widget
slider1 = Scale(tab1, variable=slider_var_length, from_=0.01, to=10, orient="horizontal", label="Time Series Length [1 for 10k points]", sliderlength=20, length=500,resolution=0.05)
slider1.bind("<Motion>", slider_Timelength_changed)
slider1.grid(row=5 ,column=3, padx=50, pady=10)
slider1.configure(bg="#5E5C5B")
#slider colors
slider1.configure(bg="#5E5C5B", troughcolor="#FF5733")
# starting value of the slider to 1
slider_var_length.set(1.0)


def slider_noise_changed(event):
    valuenoise = slider_var_noise.get()
    
#variable to be controlled by the slider
slider_var_noise = tk.DoubleVar()


slider2 = Scale(tab1, variable=slider_var_noise, from_=0, to=5, orient="horizontal", label="Standard Deviation of Gaussian Noise", sliderlength=20, length=300,resolution=0.01)

slider2.bind("<Motion>", slider_noise_changed)
slider2.grid(row=3 ,column=0, padx=20, pady=10)
slider2.configure(bg="#5E5C5B")
#  slider colors
slider2.configure(bg="#5E5C5B", troughcolor="#FF5733")
# starting value of the slider to 1
slider_var_noise.set(1.0)



def create_folders():
    try:
        selected_directory = filedialog.askdirectory(title="Select a location to create folders")
        if not selected_directory:
            messagebox.showinfo("Info", "Folder creation canceled")
            return
        
        folder_names = ["duffing", "lorenz", "noise", "rossler", "Chen"]
        subfolder_names = ["chaotic excel", "chaotic plots", "periodic plots", "periodic excel"]
        
        for folder_name in folder_names:
            folder_path = os.path.join(selected_directory, folder_name)
            os.mkdir(folder_path)
            
            if folder_name == "Chen":
                os.mkdir(os.path.join(folder_path, "hyperchaos plots"))
                os.mkdir(os.path.join(folder_path, "hyperchaos time series"))
            elif folder_name == "noise":
                os.mkdir(os.path.join(folder_path, "noise plots"))
                os.mkdir(os.path.join(folder_path, "noise time series"))
            else:
                for subfolder_name in subfolder_names:
                    subfolder_path = os.path.join(folder_path, f"{folder_name} {subfolder_name}")
                    os.mkdir(subfolder_path)
        
        messagebox.showinfo("Success", "Folders created successfully")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")





# TIME SERIES AND RECURRENCE PLOTS START

def generate_graphs():
    import pyrqa.settings as settings
    from kneed import KneeLocator
    # rho varied, sigma and beta constant
    if case == 1:
        # Parameters for Lorenz system as given in the Wikipedia page
        sigma = 10
        beta = 8/3

        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")


        rho_values = np.linspace(A, B, noofplots)  # generate  values between A and B

        # INITIAL CONDITIONS:
        tL_span = [0, 200*valuelength]  # Starting and Ending time for Lorenz system
        IL0 = [1, 1, 1]
        prev_end_point = IL0
        

        def lorenz_system(tL, XL, sigma, beta, rho):
            x, y, z = XL
            dx2_dt = sigma * (y - x)
            dy2_dt = x * (rho - z) - y
            dz2_dt = x * y - beta * z
            return [dx2_dt, dy2_dt, dz2_dt]


        time_series_data = []

        # Loop through rho values
        for rho in rho_values:
            IL0 = prev_end_point
            
            if solver_choice == 100:
                method = 'DOP853'
            elif solver_choice == 200:
                method = 'RK45'
            print(method)
            


            # Solve the ODE using the selected method
            solL = solve_ivp(lorenz_system, tL_span, IL0, args=(sigma, beta, rho),method=method,  max_step=0.01)

            
            xL1 = solL.y[0]
            yL1 = solL.y[1]
            zL1 = solL.y[2]
            tL1 = solL.t

            numx1 = len(xL1)
            print(numx1)# initial length of time series
            print("Length of original Time series , " + str(numx1))
            

            # Calculate the number of points to remove (50% of the total points)
            num_points_to_remove = int(0.5 * numx1)

            # Remove 50% of the first points from x and v
            xL = xL1[num_points_to_remove:]
            yL = yL1[num_points_to_remove:]
            zL = zL1[num_points_to_remove:]
            tL = tL1[num_points_to_remove:]



            if noise_value1 == 1: #to add nosise into the System if radio button is checked.
                mean = 0  # Mean of the Gaussian distribution
                noise = np.random.normal(mean, valuenoise, len(xL))
                xL = xL + noise
                yL = yL + noise
                zL = zL + noise

            prev_end_point = [xL[-1], yL[-1], zL[-1]] #for transient removal

            numx = len(xL)
            # Final length of time series after transient removal.

            print("Length of Time series after transient removal, " + str(numx))


            # Normalize the data components
            xL = scaler.fit_transform(xL.reshape(-1, 1)).flatten()
            yL = scaler.fit_transform(yL.reshape(-1, 1)).flatten()
            zL = scaler.fit_transform(zL.reshape(-1, 1)).flatten()




            time_series_data.append(pd.DataFrame({'time': tL, 'x': xL, 'y': yL, 'z': zL}))

            # Phase trajectory
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(xL, yL, zL, color='magenta')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title(f'Phase trajectory for Lorenz system with rho={rho}', color='magenta')
            plt.savefig(f'{save_pathplots}/Phase_rho_{rho:.10f}.png')
     
            # Time Series 
            fig = plt.figure()
            plt.plot(tL, xL, color='green', label='x')
            plt.plot(tL, yL, color='blue', label='y')
            plt.plot(tL, zL, color='red', label='z')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Lorenz system time series with rho={rho}, sigma={sigma}, beta={beta}', color='magenta')
            plt.savefig(f'{save_pathplots}/time_series_rho_{rho:.10f}.png')
     
            
            if delaymethod == 10000:
                #  list to store the average mutual information values
                ami_list = []

                #list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, xL):
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(xL, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(xL.reshape(-1, 1), xB)
                    
                    return tau, mi[0]

                start_time = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, xL) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tauL= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                fig = plt.figure()
                # Plot the average mutual information vs tau
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                plt.savefig(f'{save_pathplots}/average_mutual_information_vs_tau_rho_{rho:.10f}.png')

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time))

                
      
                
  
                
            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(xL):
                    N = len(xL)
                    mean = np.mean(xL)
                    acf = np.correlate(xL - mean, xL - mean, mode='full') / (N * np.var(xL))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(xL)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tauL=r_delay
                print(tauL)
                fig = plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                plt.savefig(f'{save_pathplots}/autocorrelation_rho_{rho:.10f}.png')
                fig = plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(xL[:-r_delay], xL[r_delay:])
                plt.savefig(f'{save_pathplots}/autocorrelation_reconstrcuted_rho_{rho:.10f}.png')
           
            
        
  
            x=xL
            tau = tauL # Embedding delay

            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15       
            Atol = 2    
            

            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')


            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])
            mL=dim[knee_point]
            
            
            
            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std_x = np.std(xL)
                epsL = stddvofsignal * std_x
                
            elif rr_stddev==20:
                data_points=xL
                m=mL
                tau=tauL
        
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-4
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                # Use the found threshold for your RQA analysis
                rqa_settings = settings.Settings(
                    time_series,
                    analysis_type=Classic(),
                    neighbourhood=FixedRadius(threshold),
                    similarity_measure=EuclideanMetric(),
                )

                computation = RQAComputation.create(rqa_settings, verbose=False)
                result = computation.run()

                computation = RPComputation.create(rqa_settings)
                result2 = computation.run()

                RR = result.recurrence_rate
                print(RR)
                epsL=threshold
                print("Theshold using % of RR, " + str(epsL))
            
 

        
            # Create TimeSeries object
            time_series = TimeSeries(xL, embedding_dimension=mL, time_delay=tauL)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsL),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_rho_{rho:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)




            # Create a new array for the reconstructed phase space
            x_reconstructed = np.zeros((mL, len(xL) - (mL - 1) * tauL))

            # Fill the array with delayed values of xD
            for i in range(mL):
                x_reconstructed[i] = xL[i * tauL: i * tauL + len(x_reconstructed[i])]

            # Plot the reconstructed phase space
            plt.figure()
            plt.plot(x_reconstructed[0], x_reconstructed[1])
            plt.xlabel('x(t)')
            plt.ylabel('x(t + tauL)')
            plt.title(f'Reconstructed phase space with rho={rho}')
            plt.savefig(f'{save_pathplots}/reconstrcuted_rho_{rho:.10f}.png')


            prev_end_point = [xL[-1], yL[-1], zL[-1]]

        # Save time series data to Excel files
        for i, rho in enumerate(rho_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_rho_{rho:.10f}.xlsx', index=False)
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete

        
   

        
   
           
    #3 Cases of keeping one parameter varying and changing the two other for rossler system
    if case == 4:
        #c varied , a and b constant

        a = 0.2
        b = 0.2

        c_values = np.linspace(A, B, noofplots)  # generate values between A and B

        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")
    
        #ROSSLER ATTRACTOR Initial conditions [X0,Y0,Z0]
        IR0 = [1,1,1]
        prev_end_point = IR0
        #Starting time and Ending time  for Rossler attractor
        tR_span = [0,800*valuelength]

        # Defining the Rossler system of differential equations
        def Rossler_system(tR, XR, a, b,c):
            x,y,z = XR
            dx1_dt = -y-z
            dy1_dt = x + a*y
            dz1_dt = b+ z*(x-c)
            return [dx1_dt, dy1_dt, dz1_dt]
        


        time_series_data = []

        for c in c_values:
            

            IR0 = prev_end_point
            
            if solver_choice == 100:
                method = 'DOP853'
            elif solver_choice == 200:
                method = 'RK45'
            print(method)

            # Solve the ODE using the selected method
            solR = solve_ivp(Rossler_system, tR_span, IR0, args=(a, b,c), method=method, max_step=0.04)
            
            

            # Extract the solutions for ROSSLER ATTRACTOR DIFFERENTIAL equations
            xR1=solR.y[0]
            yR1=solR.y[1]
            zR1=solR.y[2]
            tR1=solR.t
            
            numx1= len(xR1) 
            print("Length of original Time series , " + str(numx1))
            
            # Calculate the number of points to remove (50% of the total points)
            num_points_to_remove = int(0.5 * numx1)
            
            
     
            xR = xR1[num_points_to_remove:]
            yR = yR1[num_points_to_remove:]
            zR = zR1[num_points_to_remove:]
            tR = tR1[num_points_to_remove:]
            
            numx= len(xR)
            print("Length of Time series after transient removal, " + str(numx))
      
            prev_end_point = [xR[-1], yR[-1], zR[-1]]

            
            if noise_value1 == 1:
                mean = 0  # Mean of the Gaussian distribution
                noise = np.random.normal(mean, valuenoise, len(xR))
                xR = xR + noise
                yR = yR + noise
                zR = zR + noise
                
            
                
            # Normalize the data components
            xR = scaler.fit_transform(xR.reshape(-1, 1)).flatten()
            yR = scaler.fit_transform(yR.reshape(-1, 1)).flatten()
            zR = scaler.fit_transform(zR.reshape(-1, 1)).flatten()
            
            numx1= len(xR)
            print(numx1)
                
            time_series_data.append(pd.DataFrame({'time': tR, 'x': xR, 'y': yR, 'z': zR}))

            

            #Phase Trajectory 
            fig= plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(xR,yR,zR, color='blue')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title(f'Phase trajectory for Rossler system with c={c}', color='blue')
            plt.savefig(f'{save_pathplots}/Phase_c_{c:.10f}.png')
           
            #Time series
            fig = plt.figure()
            plt.plot(tR, xR, color='green', label='x')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Rossler system time series with a={a}, b={b}, c={c}',color='magenta')
            plt.savefig(f'{save_pathplots}/timeseries_c_{c:.10f}.png')
          
            
            
            if delaymethod == 10000:
                # Initialize an empty list to store the average mutual information values
                ami_list = []

                # Initialize an empty list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, xR):
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(xR, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(xR.reshape(-1, 1), xB)
                    
                    return tau, mi[0]

                start_time = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, xR) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tauR= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                # Plot the average mutual information vs tau
                fig = plt.figure()
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                plt.savefig(f'{save_pathplots}/average_mutual_information_vs_tau_c_{c:.10f}.png')

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time))
                
                

                
                
            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(xR):
                    N = len(xR)
                    mean = np.mean(xR)
                    acf = np.correlate(xR - mean, xR - mean, mode='full') / (N * np.var(xR))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(xR)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tauR=r_delay
                print(tauR)
                plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                plt.savefig(f'{save_pathplots}/autocorrelation_c_{c:.10f}.png')
                plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(xR[:-r_delay], xR[r_delay:])
                plt.savefig(f'{save_pathplots}/autocorrelationreconstrcuted_c_{c:.10f}.png')
            
            
            x=xR
            tau = tauR # Embedding delay
            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15       
            Atol = 2    
            

            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')

            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])

             

            mR=dim[knee_point]
         

            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std_x = np.std(xR)
                epsR = stddvofsignal * std_x
                
            elif rr_stddev==20:
                data_points=xR
                m=mR
                tau=tauR
        
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-4
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                # Use the found threshold for your RQA analysis
                rqa_settings = settings.Settings(
                    time_series,
                    analysis_type=Classic(),
                    neighbourhood=FixedRadius(threshold),
                    similarity_measure=EuclideanMetric(),
                )

                computation = RQAComputation.create(rqa_settings, verbose=False)
                result = computation.run()

                computation = RPComputation.create(rqa_settings)
                result2 = computation.run()

                RR = result.recurrence_rate
                print(RR)
                epsR=threshold
                print("Theshold using % of RR, " + str(epsR))
            
            
            

        
            # Create TimeSeries object
            time_series = TimeSeries(xR, embedding_dimension=mR, time_delay=tauR)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsR),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_c_{c:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)
            

            
            # Create a new array for the reconstructed phase space
            x_reconstructed = np.zeros((mR, len(xR) - (mR - 1) * tauR))

            # Fill the array with delayed values of xD
            for i in range(mR):
                x_reconstructed[i] = xR[i * tauR: i * tauR + len(x_reconstructed[i])]

            # Plot the reconstructed phase space
            fig = plt.figure()
            plt.figure()
            plt.plot(x_reconstructed[0], x_reconstructed[1], 'b.')
            plt.xlabel('x(t)')
            plt.ylabel('x(t + tauR)')
            plt.title(f'Reconstructed phase space with c={c}')
            plt.savefig(f'{save_pathplots}/reconstrcuted_c_{c:.10f}.png')
      

        # Save time series data to Excel files
        for i, c in enumerate(c_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_c_{c:.10f}.xlsx', index=False)
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete
        
            
        

    elif case==7:

        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")
        
        # Define parameters
        delta = 0.5  # damping coefficient
        a_values = np.linspace(A, B,noofplots)
        omega=1

        # Define Duffing equation as a system of first-order ODEs
        def duffing(t, y,a,delta,omega):
            x, v = y  # displacement and velocity
            dxdt = v  # first equation
            dvdt = x -  x**3 - delta * v + a * np.sin(omega * t)    # second equation
            return [dxdt, dvdt]
        
        
        
        def create_recurrence_plot(x, tauD, mD, epsD):
        
            N = len(x)
            num_vectors = N - (mD - 1) * tauD
            
            # Create trajectory matrix
            trajectory_matrix = np.zeros((num_vectors, mD))
            for i in range(num_vectors):
                trajectory_matrix[i] = x[i:i+mD*tauD:tauD]
            
            # Calculate pairwise distances
            pairwise_distances = np.sqrt(np.sum((trajectory_matrix[:, np.newaxis] - trajectory_matrix)**2, axis=2))
            
            # Apply recurrence threshold
            recurrence_matrix = pairwise_distances < epsD
            return recurrence_matrix
        
        

        # Define time span and initial conditions
        t_span = [0, 800*valuelength]  # time interval
        ID0 = [x,v]=[1, 1]  # initial displacement and velocity
        prev_end_point = ID0
        
        time_series_data = []

        for a in a_values:
            prev_end_point = ID0
  
            IL0 = prev_end_point
            
            if solver_choice == 100:
                method = 'DOP853'
            elif solver_choice == 200:
                method = 'RK45'
            print(method)

            # Solve the ODE using the selected method
            sol = solve_ivp(duffing, t_span, ID0, args=(a,delta,omega), method=method, max_step=0.04)
           
            
            x1 = sol.y[0]
            v1 = sol.y[1]
            tD1 = sol.t
            
            numx1=len(x1)
            print("Length of original Time series , " + str(numx1))
            # Calculate the number of points to remove (50% of the total points)
            num_points_to_remove = int(0.5 * numx1)
            
            
            
            
            # Remove the first  points from x ,t and v
            x = x1[num_points_to_remove:]
            v = v1[num_points_to_remove:]
            tD = tD1[num_points_to_remove:]
            
            numx=len(x)
            print("Length of Time series after transient removal, " + str(numx))
            
            if noise_value1 == 1: #to add nosise into the System if radio button is checked.
                mean = 0  # Mean of the Gaussian distribution
                noise = np.random.normal(mean, valuenoise, len(x))
                x = x1 + noise
                v = v1 + noise
              
          
            prev_end_point = [x[-1], v[-1]]
            


            
            # Normalize the data components
            x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            v = scaler.fit_transform(v.reshape(-1, 1)).flatten()

            
            

            # Phase trajectory for Duffing system
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, v, color='magenta')
            ax.set_xlabel('x')
            ax.set_ylabel('v')
            plt.title(f'Phase Space for Duffing system with a={a}', color='magenta')
            plt.savefig(f'{save_pathplots}/phasespace_a_{a:.10f}.png')
           

            # Time series
            fig = plt.figure()
            plt.plot(tD, x, color='green', label='x')
            plt.plot(tD, v, color='blue', label='v')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'Duffing system time series with a={a}, delta={delta}, omega={omega}', color='magenta')
            plt.savefig(f'{save_pathplots}/timeseries_a_{a:.10f}.png')
            
            time_series_data.append(pd.DataFrame({'time': tD, 'x': x, 'y': v}))
            
            
            if delaymethod == 10000:
                # Initialize an empty list to store the average mutual information values
                ami_list = []

                # Initialize an empty list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, x):
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(x, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(x.reshape(-1, 1), xB)
                    
                    return tau, mi[0]

                start_time = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, x) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tauD= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                # Plot the average mutual information vs tau
                fig = plt.figure()
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                plt.savefig(f'{save_pathplots}/average_mutual_information_vs_tau_a_{a:.10f}.png')

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time))
                
                

     
                
            
               
                
                
            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(x):
                    N = len(x)
                    mean = np.mean(x)
                    acf = np.correlate(x - mean, x - mean, mode='full') / (N * np.var(x))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(x)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tauD=r_delay
                print(tauD)
                fig = plt.figure()
                plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                plt.savefig(f'{save_pathplots}/autocorrelaion_a_{a:.10f}.png')
                fig = plt.figure()
                plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(x[:-r_delay], x[r_delay:])
                plt.savefig(f'{save_pathplots}/autocorrelation_reconstrcuted_a_{a:.10f}.png')
       
                

            x=x
            tau = tauD # Embedding delay          
            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15       
            Atol = 2    
            

            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')


            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])

            mD=dim[knee_point]



            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std_D = np.std(x)
                epsD = stddvofsignal * std_D
                
            elif rr_stddev==20:
                data_points=x
                m=mD
                tau=tauD
        
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-5
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                # Use the found threshold for your RQA analysis
                rqa_settings = settings.Settings(
                    time_series,
                    analysis_type=Classic(),
                    neighbourhood=FixedRadius(threshold),
                    similarity_measure=EuclideanMetric(),
                )

                computation = RQAComputation.create(rqa_settings, verbose=False)
                result = computation.run()

                computation = RPComputation.create(rqa_settings)
                result2 = computation.run()

                RR = result.recurrence_rate
                print(RR)
                
                
                epsD=threshold
                print("Theshold using % of RR, " + str(epsD))
            
            
            
            
            
        
            # Create TimeSeries object
            time_series = TimeSeries(x, embedding_dimension=mD, time_delay=tauD)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsD),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_a_{a:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)
            
            

            
            
  

            # Create a new array for the reconstructed phase space
            x_reconstructed = np.zeros((mD, len(x) - (mD - 1) * tauD))

            # Fill the array with delayed values of xD
            for i in range(mD):
                x_reconstructed[i] = x[i * tauD: i * tauD + len(x_reconstructed[i])]

            # Plot the reconstructed phase space
            fig = plt.figure()
            plt.figure(figsize=(10, 10))
            plt.plot(x_reconstructed[0], x_reconstructed[1], 'b.')
            plt.xlabel('x(t)')
            plt.ylabel('x(t + tauD)')
            plt.title(f'Reconstructed phase space with a={a}')
            plt.savefig(f'{save_pathplots}/reconstrcuted_a_{a:.10f}.png')
    

            IL0 = prev_end_point

        # Save time series data to Excel files
        for i, a in enumerate(a_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_a_{a:.10f}.xlsx', index=False)
            
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete
            
            
    elif case == 8:
        
        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")


         # Define parameters
        a = 0 # angular frequency
        omega_values = np.linspace(A, B, noofplots)

        # Define sine wave equation
        def sine_wave(t, a, omega):
            return a * np.sin(omega * t)
        
    

        # Define time span
        t_span = [0, 100]  # time interval
        time_series_data = []
        length = int(10000 * valuelength)  # Converts the result to an integer

        for omega in omega_values:
            t = np.linspace(t_span[0], t_span[1], length)  # time points
            x = sine_wave(t, a, omega)  # generate sine wave
            
            mean = 0  # Mean of the Gaussian distribution
            noise = np.random.normal(mean, valuenoise, len(x))
            x = x + noise
            
            
            
            # Normalize the data components
            x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            
            time_series_data.append(pd.DataFrame({'time': t, 'x': x}))


            numx = len(x)
            print("Length of Time series (no transient removal), " + str(numx))

            # Time series of sine wave system
            fig = plt.figure()
            plt.plot(t, x, color='green', label='x')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'NOISE time series with a={a}, omega={omega}', color='magenta')
            plt.savefig(f'{save_pathplots}/Timeseries_omega_{omega:.10f}.png')
     
            
            if delaymethod == 10000:
                
                # Initialize an empty list to store the average mutual information values
                ami_list = []

                # Initialize an empty list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, x):
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(x, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(x.reshape(-1, 1), xB)
                    
                    return tau, mi[0]

                start_time = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, x) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tauN= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                # Plot the average mutual information vs tau
                fig = plt.figure()
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                plt.savefig(f'{save_pathplots}/average_mutual_information_vs_tau_omega_{omega:.10f}.png')

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time))
                
                
                
            
                
         
                
            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(x):
                    N = len(x)
                    mean = np.mean(x)
                    acf = np.correlate(x - mean, x - mean, mode='full') / (N * np.var(x))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(x)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tauN=r_delay
                print(tauN)
                fig = plt.figure()
                plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                plt.savefig(f'{save_pathplots}/autocorrelation_omega_{omega:.10f}.png')
                fig = plt.figure()
                plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(x[:-r_delay], x[r_delay:])
                plt.savefig(f'{save_pathplots}/autocorrelationreconstrcuted_omega_{omega:.10f}.png')
           
                
                
            
            tau = tauN # Embedding delay
            print(tauN)

            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15       
            Atol = 2    
            

            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')


            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])
 
            mN=dim[knee_point]
            



            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std = np.std(x)
                epsN = stddvofsignal * std
                
            elif rr_stddev==20:
                data_points=x
                m=mN
                tau=tauN
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-5
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                
                epsN=threshold
                print("Theshold using % of RR, " + str(epsN))
            
            
            
            
            
            
            
        
            # Create TimeSeries object
            time_series = TimeSeries(x, embedding_dimension=mN, time_delay=tauN)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsN),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_omega_{omega:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)
            
            
           

            # Create a new array for the reconstructed phase space
            x_reconstructed = np.zeros((mN, len(x) - (mN - 1) * tau_min))

            # Fill the array with delayed values of x
            for i in range(mN):
                x_reconstructed[i] = x[i * tau_min: i * tau_min + len(x_reconstructed[i])]

            # Plot the reconstructed phase space
            fig = plt.figure()
            plt.figure(figsize=(10, 10))
            plt.plot(x_reconstructed[0], x_reconstructed[1], 'b.')
            plt.xlabel('x(t)')
            plt.ylabel('x(t + tau)')
            plt.title(f'Reconstructed phase space with a={a}')
            plt.savefig(f'{save_pathplots}/reconstructed_omega_{omega:.10f}.png')
         

        # Save time series data to Excel files
        for i, omega in enumerate(omega_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_omega_{omega:.10f}.xlsx', index=False)
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete
  
    elif case == 9:
        
        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")


        # Define parameters
        a = 0 # angular frequency
        omega_values = np.linspace(A, B, noofplots)

        # Define sine wave equation
        def sine_wave(t, a, omega):
            return a * np.sin(omega * t)

        
        length = int(10000 * valuelength)

        # Define time span
        t_span = [0, 100]  # time interval
        time_series_data = []
        for omega in omega_values:
            t = np.linspace(t_span[0], t_span[1], length )  # time points
            x = sine_wave(t, a, omega)  # generate sine wave
            
            mean = 0  # Mean of the Gaussian distribution
            noise = 0
            x = x + noise

            
            
            
            # Normalize the data components
            x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            
            time_series_data.append(pd.DataFrame({'time': t, 'x': x}))


            numx = len(x)
            print("Length of Time series (no transient removal), " + str(numx))

            tauN=10
            print(tauN)

            mE=3
            print(mE)

           

           

            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std = np.std(x)
                epsN = stddvofsignal * std
                
            elif rr_stddev==20:
                data_points=x
                m=mE
                tau=tauN
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-5
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                
                epsN=threshold
                print("Theshold using % of RR, " + str(epsN))
            
            

            
           
            # Create TimeSeries object
            time_series = TimeSeries(x, embedding_dimension=mE, time_delay=tauN)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsN),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_omega_{omega:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            # Set diagonal lines to 1 and non-diagonals to 0
            L=result2.recurrence_matrix_reverse
            # Assuming L is your input matrix
            L = np.array(result2.recurrence_matrix_reverse)


            # Create a mask for the diagonal and anti-diagonal elements
            mask_diagonal = np.eye(L.shape[0], dtype=bool)
            mask_antidiagonal = np.flip(mask_diagonal, axis=1)

            # Set non-diagonal elements to 0
            L[~mask_diagonal] = 0

            # Set anti-diagonal elements to 1
            L[mask_antidiagonal] = 1




            ImageGenerator.save_recurrence_plot(L, full_file_path)
            
            



            

        # Save time series data to Excel files
        for i, omega in enumerate(omega_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_omega_{omega:.10f}.xlsx', index=False)
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete
            
            
    if case == 10:
        
        root = tk.Tk()
        root.withdraw()
        # Ask for the directory for Excel files with a custom title
        save_pathexcel = filedialog.askdirectory(title="Select Directory to save Excel Files")

        # Ask for the directory for plots with a custom title
        save_pathplots = filedialog.askdirectory(title="Select Directory to save Plots")

        
        # Define the constants
        a = 35
        b = 4.9
        c = 25
        d = 5
        e = 35
        

        # Define the range of k values
        k_values = np.linspace(A, B, noofplots)
        
    

        # Time span
        tR_span = [0,60*valuelength]
        # Initial conditions
        IR0=[1.0, 1.0, 1.0, 1.0]
        prev_end_point = IR0
        

        
        # Define the system of differential equations
        def chen_system(variables, t, a, b, c, d, e, k):
            x, y, z, u = variables
            dxdt = a * (y - x) + e * y * z
            dydt = c * x - d * x * z + y + u
            dzdt = x * y - b * z
            dudt = -k * y
            return [dxdt, dydt, dzdt, dudt]


        # Initialize time_series_data as an empty list
        time_series_data = []

        # Create individual plots for each omega value
        for k in k_values:
            plt.figure(figsize=(6, 5))  # Create a new figure for each omega
            
            IR0 = prev_end_point
            
            if solver_choice == 100:
                method = 'DOP853'
            elif solver_choice == 200:
                method = 'RK45'
            print(method)
            
          
            sol = solve_ivp(lambda t, xyz: chen_system(xyz, t, a, b, c, d, e, k), tR_span, IR0, method='RK45', max_step=0.003)

            x, y, z, u = sol.y
            t = sol.t
            
            numx1= len(x)
            print(numx1)
            
            print("Length of original Time series , " + str(numx1))
            
            # Calculate the number of points to remove (50% of the total points)
            num_points_to_remove = int(0.5 * numx1)
            
            prev_end_point = [x[-1], y[-1], z[-1],u[-1]]
            
            
    
            # Remove the initial transient
            x = x[num_points_to_remove:]
            y = y[num_points_to_remove:]
            z = z[num_points_to_remove:]
            u = u[num_points_to_remove:]
            t = t[num_points_to_remove:]
           
            
            numx= len(x)
            print("Length of Time series after transient removal, " + str(numx))
            
            
            
            # Add Gaussian noise if needed (noise_value == 1)
            if noise_value1 == 1:
                mean = 0  # Mean of the Gaussian distribution
                noise = np.random.normal(mean, valuenoise, len(x))
                x = x + noise
                y = y + noise
                z = z + noise
                u = u+ noise

            
            # Normalize the data components
            x = scaler.fit_transform(x.reshape(-1, 1)).flatten()
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            z = scaler.fit_transform(z.reshape(-1, 1)).flatten()
            u = scaler.fit_transform(u.reshape(-1, 1)).flatten()
          
            
            # Store the time series data in a DataFrame
            time_series_data.append(pd.DataFrame({'time': t, 'x': x, 'y': y, 'z': z,'u': u}))
            
            # Plot xy Phase Space
            fig = plt.figure()
            plt.figure(figsize=(6, 6))
            plt.plot(x, y, label=f'k = {k}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('xy Phase Space')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_pathplots}/x_vs_y_k_{k:.10f}.png')
       
            
            # Plot xz Phase Space
            fig = plt.figure()
            plt.figure(figsize=(6, 6))
            plt.plot(x, z, label=f'k = {k}', color='orange')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('xz Phase Space')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_pathplots}/x_vs_z_k_{k:.10f}.png')
        
            # Plot yz Phase Space
            fig = plt.figure()
            plt.figure(figsize=(6, 6))
            plt.plot(y, z, label=f'k = {k}', color='green')
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title('yz Phase Space')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_pathplots}/y_vs_z_k_{k:.10f}.png')
       
            
            # Plot Time Series
            fig = plt.figure()
            plt.figure(figsize=(10, 6))
            plt.plot(t, x, label=f'x, k = {k}')
            plt.plot(t, y, label=f'y, k = {k}')
            plt.plot(t, z, label=f'z, k = {k}')
            plt.plot(t, u, label=f'u, k = {k}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Time Series')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_pathplots}/Timeseries_k_{k:.10f}.png')
          


            
            
            if delaymethod == 10000:
                # Initialize an empty list to store the average mutual information values
                ami_list = []

                # Initialize an empty list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, x):
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(x, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(x.reshape(-1, 1), xB)
                    
                    return tau, mi[0]

                start_time = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, x) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tauR= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                # Plot the average mutual information vs tau
                fig = plt.figure()
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                plt.savefig(f'{save_pathplots}/average_mutual_information_vs_tau_k_{k:.10f}.png')

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time))
                

                
            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(x):
                    N = len(x)
                    mean = np.mean(x)
                    acf = np.correlate(x - mean, x - mean, mode='full') / (N * np.var(x))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(x)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tauR=r_delay
                print(tauR)
                fig = plt.figure()
                plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                plt.savefig(f'{save_pathplots}/autocorrelation_k_{k:.10f}.png')
                fig = plt.figure()
                plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(x[:-r_delay], x[r_delay:])
                plt.savefig(f'{save_pathplots}/autocorrelationreconstrcuted_k_{k:.10f}.png')

          
     
            
            tau = tauR # Embedding delay
            print(tauR)



            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15 
            Atol = 2 


            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')


            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])
            mHC1=dim[knee_point] 

            

            
        
            
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std_x = np.std(x)
                epsR = stddvofsignal * std_x
                
            elif rr_stddev==20:
                data_points=x
                m=mHC1
                tau=tauR
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-5
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                
                epsR=threshold
                print("Theshold using % of RR, " + str(epsR))
        
            # Create TimeSeries object
            time_series = TimeSeries(x, embedding_dimension=mHC1, time_delay=tauR)

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic ,
                neighbourhood=FixedRadius(epsR),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )


            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            filename = f'recurrence_plot_k_{k:.10f}.png'  # Include rho in the filename
            full_file_path = os.path.join(save_pathplots, filename)
            ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)
            



         

            # Create a new array for the reconstructed phase space
            x_reconstructed = np.zeros((mHC1, len(x) - (mHC1 - 1) * tauR))

            # Fill the array with delayed values of xD
            for i in range(mHC1):
                x_reconstructed[i] = x[i * tauR: i * tauR + len(x_reconstructed[i])]

            # Plot the reconstructed phase space
            fig = plt.figure()
            plt.figure()
            plt.plot(x_reconstructed[0], x_reconstructed[1], 'b.')
            plt.xlabel('x(t)')
            plt.ylabel('x(t + tauR)')
            plt.title(f'Reconstructed phase space with k={k}')
            plt.savefig(f'{save_pathplots}/reconstrcuted_k_{k:.10f}.png')
         
            

            # Create 3D plot for reconstructed phase space
            fig = plt.figure()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the reconstructed phase space
            ax.plot(x_reconstructed[0], x_reconstructed[1], x_reconstructed[2], 'b.')
            ax.set_xlabel('x(t)')
            ax.set_ylabel('x(t + tauR)')
            ax.set_zlabel('x(t + 2*tauR)')  # You can adjust this for more dimensions
            ax.set_title(f'Reconstructed Phase Space with k={k}')
            plt.tight_layout()
            plt.savefig(f'{save_pathplots}/reconstrcuted3d_k_{k:.10f}.png')

        # Save time series data to Excel files
        for i, k in enumerate(k_values):
            time_series_data[i].to_excel(f'{save_pathexcel}/time_series_k_{k:.10f}.xlsx', index=False)
        # Print a statement after Excel files are saved
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        print("Excel files have been saved successfully.")
        status_label.config(text="Status: Completed")  # Update the status label when the task is complete
    
            








# Create a list to store the file names
file_names = []
data_frames = []


# Function to save the igraph graph image in the same folder as the Excel files
def save_graph_image(G, folder_pathnetwork, param_type, param_value):
    #  ( Fruchterman-Reingold layout)
    layout = G.layout("fr")

    # Generate the filename with parameter type and value
    filename = f"graph_{param_type}_{param_value}.png"

  
    full_file_path = os.path.join(folder_pathnetwork, filename)

    # Plot the igraph graph and save it as an image with the custom filename
    ig.plot(G, layout=layout, bbox=(800, 600), margin=20, target=full_file_path)
    
    


# Function to calculate RQA measure and Network measures START

def rqanandnetwork():
    start_time = time.time()
    RQA_NETWORK_data = pd.DataFrame(columns=['Recurrence Rate', 'Determinism', 'Laminarity','Trapping Time','LMAX','Degree Distribution',
                                             'degree centrality', 'average path length', 'Clustering Coefficient',
                                             'Diagonal Line Entropy', 'Network Density'])
    
    folder_pathrqanetwork = filedialog.askdirectory(title="Select folder having Time series excel files")
    if folder_pathrqanetwork:
        # Iterate over each file in the folder
        for filename in os.listdir(folder_pathrqanetwork):
            if filename.endswith(".xlsx"):
                file_names.append(filename)

        # Sort the file names based on the parameter value
        file_names.sort(key=lambda x: float(x.split("_")[3][:-5]))

        # Iterate over each sorted file in the folder
        for filename in file_names:
            # Extract the parameter value and type from the filename
            param_type = filename.split("_")[2]
            param_value = filename.split("_")[3][:-5]

            # Set the column title based on the parameter type
            if param_type == "rho":
                column_title = "Rho"
            elif param_type == "sigma":
                column_title = "Sigma"
            elif param_type == "beta":
                column_title = "Beta"
            elif param_type == "a":
                column_title = "A"
            elif param_type == "b":
                column_title = "B"
            elif param_type == "c":
                column_title = "C"
            else:
                column_title = param_type.capitalize()

            # Read the time series data from the file
            file_path = os.path.join(folder_pathrqanetwork, filename)
            time_series_data = pd.read_excel(file_path)

            # Extract the 'x' values from the time series data
            x = time_series_data['x']
  
            # Normalize the data components
            x = scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()


            if delaymethod == 10000:
                
                # Initialize an empty list to store the average mutual information values
                ami_list = []

                # Initialize an empty list to store the tau values
                tau_list = []
                start_time = time.time()  # Start timing
                # Loop over different values of tau (the delay parameter)
                def calculate_mutual_info(tau, x):
    
                    # Apply the delay embedding technique to create set B as x(ti+tau)
                    xB = np.roll(x, -tau)

                    # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
                    mi = mutual_info_regression(x.reshape(-1, 1), xB.reshape(-1, 1))

                    return tau, mi[0]
    

                start_time2 = time.time()  # Start timing

                # Create a list of tau values to loop over
                tau_values = list(range(1, 150))

                # Use Parallel and delayed to parallelize the mutual information calculation
                results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, x) for tau in tau_values)

                # Extract the tau and mutual information values from the results
                tau_list, ami_list = zip(*results)

                # Convert the lists to numpy arrays
                ami_array = np.array(ami_list)
                tau_array = np.array(tau_list)

                # Find the indices of local minima
                local_min_indices = argrelextrema(ami_array, np.less)
                # Find the index of the first local minimum
                first_local_min_index = local_min_indices[0][0]
                # Print the value of tau and the average mutual information at the first local minimum
                tau_min =tau= tau_array[first_local_min_index]
                ami_min = ami_array[first_local_min_index]
                print(f"The first local minimum occurs at tau = {tau_min}")
                print(f"The average mutual information at the first local minimum is {ami_min}")

                # Plot the average mutual information vs tau
                fig = plt.figure()
                plt.plot(tau_array, ami_array, marker='o')
                plt.xlabel('tau')
                plt.ylabel('average mutual information')
                plt.title('Average mutual information function')
                end_time2 = time.time()  # End timing
                elapsed_time2 = end_time2 - start_time2  # Calculate elapsed time

                print("Time it took for avg mutual info: " + str(elapsed_time2))
                

            elif delaymethod == 20000:
                # Compute autocorrelation manually
                def autocorrelation(x):
                    N = len(x)
                    mean = np.mean(x)
                    acf = np.correlate(x - mean, x - mean, mode='full') / (N * np.var(x))
                    return acf[N - 1:]

                lag = np.arange(100)
                acf = autocorrelation(x)
                r_delay = np.argmax(acf < 1.0 / np.e)
                print(f'Autocorrelation time = {r_delay}')
                tau = r_delay
                print(tau)
                fig = plt.figure()
                plt.figure()
                plt.xlabel(r'Time delay $\tau$')
                plt.ylabel(r'Autocorrelation')
                plt.plot(lag, acf[:len(lag)])  # Plot only the relevant part of acf
                plt.plot(r_delay, acf[r_delay], 'o')  # Mark the autocorrelation time
                fig = plt.figure()
                plt.figure()
                plt.title(f'Time delay = {r_delay}')
                plt.xlabel(r'$x(t)$')
                plt.ylabel(r'$x(t + \tau)$')
                plt.plot(x[:-r_delay], x[r_delay:])
         


            # Define the parameters for FNN calculation
            dim_max = 10 # Maximum embedding dimension to consider
            Rtol = 15 
            Atol = 2 
            start_time3 = time.time() 

            def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
                def FNN(d, x, tau, Rtol, Atol):
                    def reconstruct(x, dim, tau):
                        m = len(x) - (dim - 1) * tau
                        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                    def findneighbors(rec1):
                        n_neighbors = 2
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                        distance, inn = nbrs.kneighbors(rec1)
                        return inn[:, 1], distance[:, 1]

                    rec1 = reconstruct(x[:-tau], d, tau)
                    rec2 = reconstruct(x, d + 1, tau)
                    inn, distance = findneighbors(rec1)
                    
                    Ra=np.std(x) #Size of the attractor

                    R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                    R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / Ra > Atol
                    R3 = np.bitwise_or(R1, R2) #conbination of R1 and R2 as proposed.
                    return np.mean(R1), np.mean(R2), np.mean(R3)

                fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
                return fnn_R3  

            # Calculate FNN for different dimensions in parallel
            dim = np.arange(1, dim_max + 1)
            results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

            # Extract the FNN values for each dimension
            fnn_values_R3 = np.array(results)

            # Plot FNN values
            fig = plt.figure()
            plt.plot(dim, fnn_values_R3, marker='o', linestyle='-')
            plt.xlabel('Embedding Dimension (dim)')
            plt.ylabel('FNN Ratio (R3)')
            plt.title('FNN vs. Embedding Dimension')

            # Find the knee point or point of maximum change in slope
            from kneed import KneeLocator

            knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
            knee_point = knee_locator.elbow

            # Highlight the knee point on the plot
            plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
            plt.legend()

            # The knee point and corresponding dimension
            print("Knee Point Dimension:", dim[knee_point])
            m = dim[knee_point]
            end_time3 = time.time()  # End timing
            elapsed_time3 = end_time3 - start_time3  # Calculate elapsed time
            print(elapsed_time3)
            print("Time it took for FNN: " + str(elapsed_time3))
            
            import pyrqa.settings as settings
             
            if rr_stddev==30:
                # Compute recurrence plot threshold based on standard deviation of x
                std_x = np.std(x)
                eps = stddvofsignal * std_x
                print("Threshold : " + str(eps))
                
            elif rr_stddev==20:
                data_points=x
                m=m
                tau=tau
        
                time_series = TimeSeries(data_points, embedding_dimension=m, time_delay=tau)

                # Define the desired recurrence rate (eg-10%)
                desired_recurrence_rate = fixed_RR

                # Define a function to calculate the actual recurrence rate given a threshold
                def calculate_actual_recurrence_rate(threshold):
                    rqa_settings = settings.Settings(
                        time_series,
                        analysis_type=Classic(),
                        neighbourhood=FixedRadius(threshold),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1
                    )
                    computation = RQAComputation.create(rqa_settings, verbose=False)
                    result = computation.run()
                    return result.recurrence_rate

                # Bisection algorithm to find the threshold
                tolerance = 1e-4
                lower_threshold = 0.0
                upper_threshold = np.max(data_points)
                threshold = (lower_threshold + upper_threshold) / 2

                while True:
                    actual_recurrence_rate = calculate_actual_recurrence_rate(threshold)
                    if abs(actual_recurrence_rate - desired_recurrence_rate) < tolerance:
                        break
                    elif actual_recurrence_rate < desired_recurrence_rate:
                        lower_threshold = threshold
                    else:
                        upper_threshold = threshold
                    threshold = (lower_threshold + upper_threshold) / 2


                    # Add a check to stop if the threshold values are too close without convergence
                    if (upper_threshold - lower_threshold) < tolerance:
                        print("Threshold values are too close without convergence.")
                        break


                print(threshold)
                eps=threshold
                print("Theshold using % of RR, " + str(eps))



            # Create TimeSeries object
            time_series = TimeSeries(x, embedding_dimension=m, time_delay=tau)
            

            # Configure RQA settings
            rqa_settings = settings.Settings(
                time_series,
                analysis_type=Classic(),
                neighbourhood=FixedRadius(eps),
                similarity_measure=EuclideanMetric(),
                theiler_corrector=1
            )

            # Perform RQA computation
            computation = RQAComputation.create(rqa_settings, verbose=False)
            result = computation.run()
            
            computation = RPComputation.create(rqa_settings)
            result2 = computation.run()
            # result is the output of RecurrencePlotComputation
            L  = result2.recurrence_matrix_reverse[::-1]

            # Create an igraph Graph from the adjacency matrix
            G = ig.Graph.Adjacency(L, mode='undirected')
            # Remove self-loops from the graph
            G.simplify()
          
            print("Number of nodes:", G.vcount())
            print("Number of edges:", G.ecount())

 
            # Save the igraph graph image with the custom filename
            save_graph_image(G, folder_pathrqanetwork, param_type, param_value)
            

            if recurrenceplot == 1:
                filename = f'recurrence_plot_{param_type}_{param_value}.png'  
                full_file_path = os.path.join(folder_pathrqanetwork, filename)
                ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)



            # Calculate clustering coefficient
            clustering_coefficient = G.transitivity_undirected()

            # Calculate degree distribution
            degree_sequence = G.degree()
            degree_counts = np.bincount(degree_sequence)
            degree_distribution = degree_counts / degree_counts.sum()

            degree_centrality = G.degree()

            avg_path_length = G.average_path_length()

            # Calculate diagonal line lengths
            diagonal_line_lengths = np.diff(np.where(L))

            # Calculate the probabilities of each diagonal line length
            unique_lengths, length_counts = np.unique(diagonal_line_lengths, return_counts=True)
            probabilities = length_counts / np.sum(length_counts)

            # Calculate Shannon entropy of diagonal line lengths
            diagonal_line_entropy = entropy(probabilities, base=2)

            # Calculate network density
            num_edges = G.ecount()  # Get the number of edges
            max_possible_edges = (G.vcount() * (G.vcount() - 1)) / 2  # Max possible edges in an undirected graph
            density = num_edges / max_possible_edges

            # After processing each file, create a DataFrame for the measure values
            measure_df = pd.DataFrame({
                column_title: [param_value],
                'Recurrence Rate': [result.recurrence_rate],
                'Determinism': [result.determinism],
                'Laminarity': [result.laminarity],
                'Trapping Time': [result.trapping_time],
                'LMAX': [result.longest_diagonal_line],
                'Degree Distribution': [degree_distribution],
                'degree centrality': [degree_centrality],
                'average path length': [avg_path_length],
                'Clustering Coefficient': [clustering_coefficient],
                'Diagonal Line Entropy': [diagonal_line_entropy],
                'Network Density': [density]
            })

            # Append the measure DataFrame to the list
            data_frames.append(measure_df)

        # After processing all files, concatenate the DataFrames in the list
        RQA_NETWORK_data = pd.concat(data_frames, ignore_index=True)

        export_path = os.path.join(folder_pathrqanetwork, 'RQA_NETWORK_measures.xlsx')
        RQA_NETWORK_data.to_excel(export_path, index=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
        print('RQA and network measures exported to:', export_path)
    else:
        print("No folder selected.")
 
    




def update_status(message):
    status_label12.config(text=message)
    root.update_idletasks()

def rqanetowrkMLcode():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    # Pick an Excel file
    file_path = filedialog.askopenfilename(
        title="Select Excel File with Features and added CLASS label",
        filetypes=[("Excel files", "*.xls;*.xlsx")]
    )
    if not file_path:
        print("No file selected. Exiting File Selector")
        return
    
    data = pd.read_excel(file_path)
    X = data.drop(columns=["CLASS"])  # Features
    y = data["CLASS"]  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(),
            "params": {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]},
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(),
            "params": {
                'n_estimators': [10,20,30,40,50,60,70,80,90,100,200],
                'max_samples': [0.1,0.3,0.5, 0.7, 0.9],
            },
        },
        {
            "name": "k-NN",
            "estimator": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],  # Number of neighbors to consider
                "weights": ["uniform", "distance"],  # Weight function used in prediction
                "p": [1, 2],  # Power parameter for Minkowski distance metric
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # Algorithm used to compute the nearest neighbors
                "leaf_size": [10, 20, 30, 40, 50],  # Leaf size passed to BallTree or KDTree
                # Additional distance metrics
                "metric": ["euclidean", "manhattan", "chebyshev"],
            }
    
        },
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=10000),
            "params": {"C": [0.1, 1.0, 10.0,100.0], "solver": ["liblinear", "lbfgs"]},
        },
        {
            "name": "SVM Kernels",
            "estimator": SVC(),
            "params": {
                "C": [0.1, 1, 10, 100, 1000],  # Penalty parameter C
                "gamma": [1, 0.1, 0.01, 0.001, 0.0001],  # Kernel coefficient for 'rbf'
                "kernel": ['rbf'],  # Different kernel functions

            },
        },
    

    ]

    directory = os.path.dirname(file_path)
    results_list = []

    for model in models:
        clf = GridSearchCV(model["estimator"], model["params"], cv=10)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        
        if model['name'] in ['Random Forest', 'Decision Tree']:
            if model['name'] == 'Random Forest':
                feature_importances = best_model.feature_importances_
            else:
                feature_importances = best_model.feature_importances_

            plt.figure(figsize=(8, 6))
            plt.barh(X.columns, feature_importances)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title(f'{model["name"]} - Feature Importance')
            plt.tight_layout()

            plt.savefig(os.path.join(directory, f"{model['name']}_feature_importance.png"))
            plt.close()

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        confusion = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        class_accuracies = np.diag(confusion) / np.sum(confusion, axis=1)
        
        class_labels = np.unique(y_test)  # Extract unique class labels






        results_text_file = os.path.join(directory, "model_evaluation_results.txt")

        with open(results_text_file, "a") as text_file:
            text_file.write(f"Model: {model['name']}\n")
            text_file.write(f"Best Parameters: {clf.best_params_}\n")
            text_file.write(f"Accuracy: {accuracy}\n")
            text_file.write(f"Precision: {precision}\n")
            text_file.write(f"Recall: {recall}\n")
            text_file.write(f"F1 Score: {f1}\n")
            text_file.write(f"Confusion Matrix:\n{confusion}\n")
            text_file.write(f"Per-Class Accuracy:\n")
            for i in range(len(class_accuracies)):
                text_file.write(f"{class_labels[i]}: {class_accuracies[i]}\n")
            text_file.write("Classification Report:\n")
            text_file.write(class_report)
            text_file.write("\n\n")

        results_dict = {
            "Model": model["name"],
            "Best Parameters": str(clf.best_params_),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": str(confusion),
        }

        # Adding per-class accuracies to the results_dict
        for i in range(len(class_accuracies)):
            results_dict[f"Class_{class_labels[i]}_Accuracy"] = class_accuracies[i]

        results_list.append(results_dict)

        model_filename = os.path.join(directory, f"{model['name']}_model.joblib")
        joblib.dump(best_model, model_filename)

    results_df = pd.DataFrame(results_list)
    results_file = os.path.join(directory, "model_evaluation_results.xlsx")
    results_df.to_excel(results_file, index=False)
    print(f"Results saved to {results_file}")




status_label12 = tk.Label(tab3, text="Training Program is Idle- Pick the Feature Excel File using this button. Trained Models will be saved in the same folder", padx=10, pady=10)


status_label12.configure(fg="white", bg="#1e1f26")




status_label12.grid(row=0, column=1) 





# Function to perform predictions and update the Excel file
def perform_predictions():
    # Update the status label
    status_label.config(text="Performing predictions...")

    #  pick an Excel file
    file_pathperform_predictions = filedialog.askopenfilename(
        title="Select Excel Data File with features",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if not file_pathperform_predictions:
        messagebox.showerror("Error", "No file selected. Please select an Excel file.")
        status_label.config(text="Idle")
        return

    # Load the machine learning model
    model_path = filedialog.askopenfilename(
        title="Select pre trained model file",
        filetypes=[("Joblib files", "*.joblib")]
    )

    if not model_path:
        messagebox.showerror("Error", "No model selected. Please select a saved model.")
        status_label.config(text="Idle")
        return

    # Load the data from the selected Excel file
    data = pd.read_excel(file_pathperform_predictions)
    X = data

    # Load the trained model
    try:
        loaded_model = joblib.load(model_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the model: {str(e)}")
        status_label.config(text="Idle")
        return

    # Make predictions
    predictions = loaded_model.predict(X)

    # Add the predicted class as a new column
    data['Predicted_CLASS'] = predictions

    # Save the updated data to a new Excel file
    save_path = filedialog.asksaveasfilename(
        title="Save Predictions as Excel File",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if save_path:
        data.to_excel(save_path, index=False)
        messagebox.showinfo("Success", "Predictions saved to a new Excel file.")
    
    # Update the status label
    status_label.config(text="idle")




status_label = tk.Label(tab3, text="Predictions program is Idle-Choose the excel file with features for prediction and the saved model to use. Check Console window in tab1 for any errors")

status_label.configure(fg="white", bg="#1e1f26")

status_label.grid(row=2, column=1) 




# Modify the image size
target_image_size = (500, 500)
class_mapping = {
    'PERIODIC': 0,
    'CHAOTIC': 1,
    'NOISE': 2,
    'EMPTY': 3,
    'HYPERCHAOS':4,
}

class_mapping_inv = {v: k for k, v in class_mapping.items()}

def load_model(filename):
    return joblib.load(filename)

def train_and_evaluate_classifier(root_folder,save_model_path):
    status_var.set("Running classifier...")

    def train_and_save_model():
        # Define hyperparameters
        batch_size = 10
        epochs = 5


        # Data augmentation and preprocessing for training data
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )

        # Load and preprocess data using the class mapping
        train_data = datagen.flow_from_directory(
            root_folder ,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=list(class_mapping.keys()),
            target_size=target_image_size, # Resize images to target size
            color_mode='grayscale'
        )

        validation_data = datagen.flow_from_directory(
            root_folder ,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=list(class_mapping.keys()),
            target_size=target_image_size,  # Resize images to target size
            color_mode='grayscale'
        )

        # Determine input shape based on the first batch of training data
        input_shape = train_data[0][0].shape[1:]

        # Build a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(class_mapping.keys()), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(validation_data)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")



        model_filename = os.path.join(root_folder , "model.joblib")
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")

    thread = threading.Thread(target=train_and_save_model)
    thread.start()

def run_prediction(model):
    def make_predictions():

        image_folder = filedialog.askdirectory(title="Select Image Folder")

        image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
        predictions_folder = os.path.join(image_folder, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)

        predictions = []
        

        for image_path in image_paths:
            # Load and preprocess each image
            img = Image.open(image_path)
            img = img.resize(target_image_size)  # Resize to target size
            img = np.array(img) / 255.0  # Normalize the image

            # Make predictions for the image
            prediction = model.predict(np.expand_dims(img, axis=0))
            predicted_label = list(class_mapping.keys())[np.argmax(prediction)]
            predictions.append(predicted_label)

            overlay_text = f'Predicted: {predicted_label}'

            # Create a new figure and axis
            fig, ax = plt.subplots()

            # Display the image
            ax.imshow(img)

            # Add overlay text
            ax.text(200, 200, overlay_text, color='red', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

            # Remove axes for display
            ax.axis('off')

            # Save the image with the overlay to the predictions folder
            image_filename = f"prediction_{os.path.basename(image_path)}.png"
            image_path = os.path.join(predictions_folder, image_filename)
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)

            # Close the figure 
            plt.close(fig)

        status_var.set("Predictions are done.")
        # function to create the slider
        create_slider(image_paths)

    # Create a thread for making predictions
    thread = threading.Thread(target=make_predictions)

    # Start the predictions thread
    thread.start()

size = 400, 400

# global variable for the slider
global slider

# Create a function to update the displayed image
def update_displayed_image(image_paths, index):
    if 0 <= index < len(image_paths):
        image_path = image_paths[index]
        img = Image.open(image_path)
        img.thumbnail(size, Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        display_label.config(image=img)
        display_label.image = img

def create_slider(image_paths):
    global slider
    slider_label = tk.Label(tab4, text="Select Image:")
    slider_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    slider = tk.Scale(tab4, from_=0, to=len(image_paths) - 1, orient=tk.HORIZONTAL, command=lambda value: update_displayed_image(image_paths, value))
    slider.grid(row=3, column=1, padx=10, pady=10, sticky="e")

    global display_label
    display_label = tk.Label(tab4)
    display_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    # Set the default value for the slider to 0
    slider.set(0)
    update_displayed_image(image_paths, 0)

status_var = tk.StringVar()
status_var.set("For test/train click first button and select the data file and name to save the model in the first and second file window:")

status_label2 = tk.Label(tab4, textvariable=status_var)
status_label2.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
status_label2.configure(fg="white", bg="#1e1f26")

status_var2 = tk.StringVar()
status_var2.set("For predictions select saved model and then select the folder with images to make predictions on:")
status_label3 = tk.Label(tab4, textvariable=status_var2)
status_label3.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")
status_label3.configure(fg="white", bg="#1e1f26")

def run_classifier():
    root_folder = filedialog.askdirectory(title="Select main folder with folders having different recurrence plot classes and corresponding data in them")
    save_model_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib Files", "*.joblib")])
    if root_folder and save_model_path:
        thread = threading.Thread(target=train_and_evaluate_classifier, args=(root_folder, save_model_path))
        thread.start()

def run_image_predictor():
    model_path = filedialog.askopenfilename(title="Select Pre-trained Model", filetypes=[("Joblib Files", "*.joblib")])
    if model_path:
        loaded_model = load_model(model_path)
        thread = threading.Thread(target=run_prediction, args=(loaded_model,))
        thread.start()


classifier_button = tk.Button(tab4, text="Train/Test", command=run_classifier, bg="#117777", fg="white")
classifier_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

predictor_button = tk.Button(tab4, text="Run Image Predictor", command=run_image_predictor, bg="green", fg="white")
predictor_button.grid(row=3, column=0, padx=10, pady=10, sticky="e")



############################################




############################



def set_variables():
    global case, A, B, var, noise_value1 ,stddvofsignal,noofplots,mL,mR,m,solver_choice,valuelength,valuenoise,delaymethod,fixed_RR,rr_stddev,fixed_RR# global variables
    selected_value1 = selected_option1.get()
    selected_value2 = selected_option2.get()
    selected_value4 = selected_option4.get()
    rr_stddev=rr_stddev_tab1.get()
    status_label.config(text="Status: runnning")
    noise_value1 = noise_value.get()
    valuelength = slider_var_length.get()
    valuenoise = slider_var_noise.get()
    print(f"Time series Value: {valuelength}")

    A_input = a_entry.get()
    try:
        A = float(A_input)
    except ValueError:
        A = 0  # set A to 0 if input is not a valid integer

    B_input = b_entry.get()
    try:
        B = float(B_input)
    except ValueError:
        B=0
        
    stddvofsignal_input = fixed_RR_entry_tab1.get()
    try:
        stddvofsignal = float(stddvofsignal_input)
    except ValueError:
        stddvofsignal=0
        
    fixed_RR_entry_input = fixed_RR_entry_tab1.get()
    try:
        fixed_RR = float(fixed_RR_entry_input)
    except ValueError:
        fixed_RR=0
        
    fixed_RR_entry_input = fixed_RR_entry_tab1.get()
    try:
        fixed_RR = float(fixed_RR_entry_input)
    except ValueError:
        fixed_RR =0
        
        

    if rr_stddev == "Fixed Recurrance rate":
        rr_stddev = 20
    elif rr_stddev == "Percentage of Standard deviation of signal to use as threshold":
        rr_stddev = 30
        
        
        
        
    noofplots_input = noofplots_entry.get()
    try:
        noofplots = int(noofplots_input)
    except ValueError:
        noofplots=0
    if selected_value4 == "Mutual-Information":
        delaymethod = 10000
    elif selected_value4 == "Autocorrelation":
        delaymethod = 20000
    
        

    if selected_value2 == "SOLVER:DOP853":
        solver_choice = 100
    elif selected_value2 == "SOLVER:RK45":
        solver_choice = 200
        

    if selected_value4 == "Mutual-Information":
        delaymethod = 10000
    elif selected_value4 == "Autocorrelation":
        delaymethod = 20000
        
        
    if selected_value1 == "Lorenz-rho varied, sigma and beta constant":
        case = 1
    elif selected_value1 == "Lorenz-sigma varied, rho and beta constant":
        case = 2
    elif selected_value1 == "lorenz-beta varied, sigma and rho constant":
        case = 3
    elif selected_value1 == "Rossler-c varied, a and b constant":
        case = 4
    elif selected_value1 == "Rossler-a varied, c and b constant":
        case = 5
    elif selected_value1 == "Rossler-b varied keeping a and c constant":
        case = 6
    elif selected_value1 == "Duffing-a varied ":
        case = 7
    elif selected_value1 == "Noise":
        case = 8
    elif selected_value1 == "Empty":
        case = 9
    elif selected_value1 == "Chen (hyperchaos)":
        case = 10
    else:
        print("0")

    print(f"A is {A}, B is {B}, stddvofsignal is {stddvofsignal}")
    print("This window can be used for identification of Errors and values obtained during the computation")
    print("Navigate the folder you selected to see the plots being generated. All Time series fies will be added to the respective folder at the end.")
    
    
    # Start a new thread 
    t = threading.Thread(target=generate_graphs)
    t.start()
    
    

    
#function to run both functions
def rqanetwork_recurrenceplot_functions_combined():
    global delaymethod, x, tau, m, eps,recurrenceplot,rr_stddev,stddvofsignal,fixed_RR
    recurrenceplot = recurrenceplot.get()
    selected_value4 = selected_option4.get()
    rr_stddev=rr_stddev_tab2.get()
    
    stddvofsignal_input = stddvofsignal_entry_tab2.get()
    try:
        stddvofsignal = float(stddvofsignal_input)
    except ValueError:
        stddvofsignal=0
        
        
    fixed_RR_entry_input = fixed_RR_entry_tab2.get()
    try:
        fixed_RR = float(fixed_RR_entry_input)
    except ValueError:
        fixed_RR=0
    
    if rr_stddev == "Fixed Recurrance rate":
        rr_stddev = 20
    elif rr_stddev == "Percentage of Standard deviation of signal to use as threshold":
        rr_stddev = 30
    
    

    if selected_value4 == "Mutual-Information":
        delaymethod = 10000
    elif selected_value4 == "Autocorrelation":
        delaymethod = 20000
    rqanandnetwork()
    


# Function to start both functions in a single thread
def start_both_rqanetwork_recurrenceplot_functions():
    thread = threading.Thread(target=rqanetwork_recurrenceplot_functions_combined)
    thread.start()

    
root.configure(background="#1e1f26")
# Set the title of the window
root.title("Unravelling Temportal Patterns")


#button to create folders
folderbutton = tk.Button(tab1, text="Create Folders/Subfolders to save data", command=create_folders)
folderbutton.grid(row=1, column=3, columnspan=2, padx=50, pady=20, sticky="nsew")
folderbutton.config(bg="#5E5C5B", fg="black", font=font.Font(family="Lato", size=9, weight="bold"))



# Create a button to calculate RQA measure and Network measures
rqa_button = tk.Button(tab2, text="Calculate RQA measures and Network measures", command=start_both_rqanetwork_recurrenceplot_functions, width=40)
rqa_button.grid(row=11, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
rqa_button.config(bg="#ff5737", fg="black", font=font.Font(family="Lato", size=9, weight="bold"))


# Create a button to run analysis
buttonML = tk.Button(tab3, text="Train/Test", command=rqanetowrkMLcode)
buttonML.config(bg="#117777", fg="black", font=("Lato", 9, "bold"))
buttonML.grid(row=1, column=1, padx=20, pady=20) 

# Create a button for prediction process
predict_button = tk.Button(tab3, text="Perform Predictions", command=perform_predictions)
predict_button.config(bg="green", fg="black", font=("Lato", 9, "bold"))
predict_button.grid(row=3, column=1, padx=22, pady=22) 



# Create a Text widget for displaying console output in tab1
output_text = tk.Text(tab1, wrap=tk.WORD, bg='#1e1f26', fg='white',width=70, height=20)  # Set background to black and foreground (text) to white
output_text.grid(row=15, column=1, padx=3, pady=4)

output_label = tk.Label(tab1, text="Console Window:")
output_label.grid(row=15, column=0, columnspan=1, padx=5, pady=5, sticky="nsew")
output_label.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=9, weight="bold"))




button = tk.Button(tab1, text="RUN- Generate the recurrence plots and time series data", command=set_variables, width=20)
button.grid(row=10, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
button.config(bg="#2E778C", fg="black", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a label to display the status
status_label = tk.Label(tab1, text="Status: Idle")
status_label.grid(row=12, column=0, columnspan=1, padx=5, pady=5, sticky="nsew")
status_label.config(bg="#34495E", fg="black", font=font.Font(family="Lato", size=9, weight="bold"))

# Create a button to stop the program 
stop_button = tk.Button(tab1, text="Stop Program", command=root.destroy)
stop_button.grid(row=12, column=1, columnspan=1, padx=3, pady=4, sticky="nsew")
stop_button.config(bg="#AD2929", fg="white", font=font.Font(family="Lato", size=9, weight="bold"))

root.protocol("WM_DELETE_WINDOW", root.destroy)



label_text = """

This project encompasses a program featuring a graphical user interface (GUI) designed to perform the following tasks:

        1) Generate Time Series Data: Create time series data in Excel files and generate corresponding Recurrence Plot images, 
           Phase Space plots, and Reconstructed Phase Space plots.

        2) Analyze Time Series Data: Conduct Recurrence Quantification Analysis (RQA) and Network Analysis on the generated data.

        3) Train and evaluate Machine Learning Models to classify time series data (PERIODIC, CHAOTIC, NOISE, HYPERCHAOS) using 
           two data formats: RQA (Recurrence Quantification Analysis) + Network Features and Recurrence Plot Images.

        4) Predictions on Unseen Data: Utilize the trained models to make predictions on unseen data.

        5) General Machine Learning Tasks: Tab 3 and Tab 4 can be employed for general machine learning classification and 
         prediction tasks as well.
    
    
    
For more documentation use the link-   https://github.com/am0032/Unravelling-Temporal-Patterns/tree/main
    













Author:
Athul Mohan
   
The Program was developed during my project work under the guidance of Professor:
Chandrakala Meena
Indian Institute of Science Education and Research
TVM

"""


labelabout = tk.Label(tab5, text=label_text, justify="left")
labelabout.grid(column=0, row=0)
labelabout.grid(row=15, column=0, columnspan=1, padx=5, pady=5, sticky="nsew")
labelabout.config(bg="#1e1f26", fg="#8c9da9", font=font.Font(family="Lato", size=12, weight="bold"))


import tkinter as tk
import math

# Parameters for the sine wave
amplitude = 0
frequency = 0.08  # Adjust this for wave speed
offset = 25

def draw_sine_wave(canvas, time):
    global noise_value1
    noise_value1 = noise_value.get()
    selected_value1 = selected_option1.get()
    canvas.delete("wave")  
    valuenoise = slider_var_noise.get()
    
    
    if noise_value1 == 1:  # Check if noise is enabled
     for x in range(0, 500, 1):  # Adjust the range and step size
         y = amplitude * math.sin(frequency * x + time) + offset
         mean = 0  
         noise = np.random.normal(mean, valuenoise)
         y = y + noise
        
         canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="white", tags="wave")
         
    
    elif selected_value1 == "Noise":
        for x in range(0, 500, 1):  # Adjust the range and step size
            y = amplitude * math.sin(frequency * x + time) + offset
            mean = 0  
            noise = np.random.normal(mean, valuenoise)
            y = y + noise
           
            canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="white", tags="wave")

       
    canvas.after(50, draw_sine_wave, canvas, time + 1)


canvas = tk.Canvas(tab1, width=500, height=50, borderwidth=0, highlightthickness=0)
canvas.grid(row=3, column=1, padx=0, pady=0)
canvas.configure(bg="#1e1f26")

draw_sine_wave(canvas, 0)



root.mainloop()

