#importing python libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import statistics as stat

#defining gaussian function as shown in lecture
def fit_func(x,a,mu,sig,m,c):
    gaussian = a*np.exp(-(x-mu)**2/(2*sig**2))
    line = m*x + c
    return gaussian + line

#importing data
spec_data = np.loadtxt(r'CompProjectData/Halpha_spectral_data.csv', skiprows=4, delimiter=",")   #loading spectral data from excel 
distance_data = np.loadtxt(r'CompProjectData/Distance_Mpc.txt', skiprows=1, delimiter="\t")   #loading distance data containing valid and invalid data

#constants
c = 2.99792458e8  #speed of light in m/s ref:https://www.britannica.com/science/speed-of-light
H_alpha_spect_line = 656.28e-9 #the red visible spectral line created by a hydrogen atom when an electron falls from the third lowest to second lowest energy level ref:https://ismlandmarks.wordpress.com/h-alpha-emission/


#removing distance data with invalid instrument response (also removed tabs from data manually)
ord_dist_dat=[]   #array created for data with valid response (3rd column equal to 1) so that values can be appended (ordered)

#for loop checks the third column of each row for a value of 1 and appends that row to the created array if 
for i in range(len(distance_data)):    
    if distance_data[i,2]==1:
        ord_dist_dat.append(distance_data[i])

#collection of values extracted from provided data
frequency_data = spec_data[0,1:] # selects first row excluding the first column

n = 25  #observation number - 'hard' coded to simplify code                          

ord_dist_dat_mat =np.reshape(ord_dist_dat,(n,3)) #converts an array of arrays into matrix which is much easier to manipulate (at least to my knowledge)

ord_dist_dat_mat = ord_dist_dat_mat[np.argsort(ord_dist_dat_mat[:,0])] #ordering both the distance and spectral data 
spec_data = spec_data[np.argsort(spec_data[:,0])]                      #in terms of ascending observation number so corresponding values can be compared

ord_spec_data = []      #will become a matrix of all the relevant intensity values

for i in range(n):      # for loop determines which data values have the same observation number  
    t = int(np.where(spec_data[:,0] == ord_dist_dat_mat[i,0])[0]) 
    for j in range(1001):
        ord_spec_data.append(spec_data[t,j])   #ord_dist_dat_mat observation numbers are all valid and in ascending order and so the intensity data will be appended in order

#creating variables that will simplify the code    
ord_spec_data =  np.reshape(ord_spec_data,(25,1001)) # there are 1001 data points for each observation number
frequency_data = spec_data[0,1:]

#velocity array created
v_data = []

#this for loop will: plot 25 graphs of intensity against frequency; produce initial values for the gaussian function and calculating the 'red-shift' velcoity for each observation 
for i in range(n):
    
    #plotting intensity data
    plt.subplot(5,5,i+1)   
    plt.plot(frequency_data, ord_spec_data[i,1:])
    
    #using polyfit function for data to find intial values of gradient and y-intercept
    p_nomial = np.polyfit(frequency_data,ord_spec_data[i,1:],1)
    m_ini = p_nomial[0]   #estimate of gradient
    c_ini =p_nomial[1]    #estimate of y-inercept

    #finding amplitude and average by finding the point of maximum difference between the intensity values and the line
    displacement=ord_spec_data[i,1:] - (m_ini*frequency_data + c_ini) 
    max_diff=np.where(displacement == max(displacement)) 
   
    #initial amplitude
    a_ini = max((ord_spec_data[i,1:]))-(m_ini*spec_data[0,int(max_diff[0])]+c_ini) #inserting max_diff back into diff
    
    #mean can also be approximated to the frequency value at the max imum difference
    mu_ini = frequency_data[int(max_diff[0])]

    #only way I could find on my own to claculate the standard deviation
    sig_ini = stat.stdev(frequency_data)

    #mean can be approximated around the point of maximum difference
    mu_ini = frequency_data[int(max_diff[0])]
    
    #array of initial values
    value_ini = [a_ini,mu_ini,sig_ini,m_ini,c_ini]
    
    #using the initial values to produce the gaussian fit
    params,params_cov = curve_fit(fit_func,frequency_data,ord_spec_data[i,1:],value_ini,maxfev=1849594)  
    
    #plotting fit
    plt.subplot(5,5,i+1) 
    plt.plot(frequency_data,fit_func(frequency_data,params[0],params[1],params[2],params[3],params[4])) #these parameters are the computers value for the variables

    #using wavespeed equation to determine wavelength - params[1] = mean frequency
    wavelength = (c/params[1])

    #finding the velocity by rearranging the provided doppler shift equation
    obs_ex_wav=(wavelength/H_alpha_spect_line)     #ratio between observed and expected wavelngth                                       
    v = (c*(((obs_ex_wav**2)-1)/((obs_ex_wav**2)+1)))/1000   #rearranged equation divided by 1000 to convert into km/s
    v_data.append(v)

plt.show()

#plotting velocity against distance set up graph
plt.figure()
plt.title('velocity against distance')
plt.ylabel('Redshift Velocity/(km/s)')
plt.xlabel('Distance/MPc') 

#using polyfit function to fit line to data and determine its uncertainty
poly_con, cov_poly_con = np.polyfit(ord_dist_dat_mat[:,1],v_data,1,cov=True)
polynomial_con = np.poly1d(poly_con)
con_unc = np.sqrt((np.abs(cov_poly_con[0,0])))  

#plotting the linear function and the points
plt.plot(ord_dist_dat_mat[:,1],polynomial_con(ord_dist_dat_mat[:,1]))            
plt.plot(ord_dist_dat_mat[:,1],v_data,'x')                                      

plt.show()                                                                                  

hubble_constant = poly_con[0] #hubble constant is equal to the gradient

#print reults
print(hubble_constant,"±",con_unc,"km/s/Mpc")       

