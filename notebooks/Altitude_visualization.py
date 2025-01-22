#!/usr/bin/env python
# coding: utf-8

# # Visualization of different altitude sensor measurements 
# I need understand what each value is

# ## Plotting altitude values 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cwd = os.getcwd()

#example parquet fdir for plotting
fdir = 'Tail_687_1_parquet'

#example parquet file
file = '687200104111158.parquet'
#file = '687200104202027.parquet'
#file = '687200107122323.parquet'
#file = '687200104162039.parquet'
#file = '687200104181334.parquet' #bad file?
#file = '687200108171530.parquet'
#file = '687200107121151.parquet'
#file = '687200109061936.parquet'
#file = '687200109110853.parquet'

#files used in the NN
#file = "687200107192334.parquet" #late start on the recording
#file = "687200107301239.parquet"
#file = "687200104261527.parquet"
#file = "687200107251002.parquet"
#file = "687200104301119.parquet"
#file = "687200107101600.parquet" #max alt 5380 instead of the regular alt 5500
#file = "687200104170717.parquet"
#file = "687200107181544.parquet"
#file = "687200104202027.parquet"
#file = "687200107170234.parquet"
#file = "687200107251652.parquet"
#file = "687200107122323.parquet" #late start to recording
#file = "687200104162039.parquet"
#file = "687200107311025.parquet"
#file = "687200104181334.parquet"
#file = "687200107171131.parquet"
#file = "687200104181127.parquet"
#file = "687200107241524.parquet"
#file = "687200107060930.parquet"
#file = "687200107150546.parquet"

pname = os.path.join(cwd,fdir,file)

df = pd.read_parquet(path=pname)
#df['ALT']
df


# In[3]:


#output value to model with NN
Tlist = ['ALT']

#list of what I think are the dependent variables to create a model for T.
Xlist = ['time', 
         'RALT', 
         'ALTR', 'IVV', 'VSPS', 
         'VRTG', 'LATG', 'LONG', 'FPAC', 'BLAC', 'CTAC', 
         'PSA', 'PI', 'PT', 
         'TAS', 'CAS', 'GS', 'CASS', 'WS', 
         'PTCH', 'ROLL', 'DA', 
         'TAT', 'SAT', 
         'LATP', 'LONP']

##new list of what I think are the dependent variables to create a model for T.
#Xlist = ['time', 
#         'RALT', 
#         'PSA', 'PI', 'PT', 
#         'ALTR', 'IVV', 
#         'VRTG', 'LATG', 'FPAC', 'BLAC', 'CTAC', 
#         'TAS', 'CAS', 'GS', 'CASS', 'WS', 'PTCH', 'ROLL', 'DA', 'TAT', 
#         'SAT', 'LATP', 'LONP']

#may want to try this with Selected Alt (ALTS) in the Xlist


# In[4]:


#altitude sensor measurements
#pressure altitude
alt  = df['ALT']
#selected altitude
alts = df['ALTS']
#radio altitude
ralt = df['RALT']
#barometric corrected altitude
baro1= df['BAL1']
#barometric corrected altitude
baro2= df['BAL2']

#----
#average static pressure
psa= df['PSA']
#Impact pressure (dynamic pressure?)
pi = df['PI']
#total pressure
pt = df['PT']

#static pressure (inches of mercury converted to mB)
ps = df['PS'] * 33.86


#---
#altitude rate
altr = df['ALTR']
#Inertial verticle speed
ivv = df['IVV']
#Selected verticle speed
vsps = df['VSPS']

#---
#verticle acceleration
vertg = df['VRTG']
latg  = df['LATG']
long  = df['LONG']
fpac  = df['FPAC']
blac  = df['BLAC']
ctac  = df['CTAC']


#calt = df['CALT']

time = df['time']


# In[5]:


np.min(ralt), np.max(ralt)


# In[8]:


plots4paper = True
singlecolumn = True

if plots4paper:
    #----------------smaller plots for the paper-----------

    #if the paper i'm making it for is single column make the plot wider
    if singlecolumn:
        fig = plt.figure(figsize=(30, 15))
    else:
        fig = plt.figure(figsize=(15, 15))
        
    #fig.suptitle('Flight Values Comparison Plot',fontsize=20)

    legendfont = 16


    #number of vertical subplots
    nv = 4

    #------
    #subplot 1
    ax1 = plt.subplot(nv,1,1)
    plt.plot(time, alt,  'r.', label='Pressure Altitude "truth"')
    #plt.plot(time, baro1,'b.', label='Baro Correct Altitude (BAL1) 4Hz')
    plt.plot(time, ralt, 'g.', label='Radio Altitude ')
    #plt.plot(time, baro2,'g-', label='Baro Correct Altitude (BAL2)')
    #plt.plot(time, alts, 'y.', label='Selected Alt (ALTS)')
    ax1.set_title('(a) Altitude', fontsize=20, fontweight="bold")
    plt.ylabel('Altitude (FT)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='upper left');

    #------
    #subplot 2
    ax2 = plt.subplot(nv,1,2)
    plt.plot(time, altr,'r.', label='Altitude Rate')
    plt.plot(time, ivv, 'b.', label='Inertial Vert Speed')
    plt.plot(time, vsps,'y.', label='Selected Vert Speed')
    ax2.set_title('(b) Altitude Rate', fontsize=20, fontweight="bold")
    plt.ylabel('Altitude Rate (FT/Min)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='upper left');

    #------
    #subplot 3
    ax3 = plt.subplot(nv,1,3)
    plt.plot(time, vertg,'r.', label='Vertical Accel')
    plt.plot(time, latg, 'y.', label='Lateral Accel')
    plt.plot(time, long, 'k.', label='Longitudinal Accel')
    #plt.plot(time, fpac, 'b.', label='Flight Path Acceleration (FPAC)')
    #plt.plot(time, blac, 'c.', label='Body Long Acceleration (BLAC)')
    #plt.plot(time, ctac, 'g.', label='Cross Track Acceleration (CTAC)')
    ax3.set_title('(c) Earth Axis Accel', fontsize=20, fontweight="bold")
    plt.ylabel('Acceleration (G)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='lower left');

    #------
    #subplot 4
    ax4 = plt.subplot(nv,1,4)
    plt.plot(time, fpac, 'b.', label='Flight Path Accel')
    plt.plot(time, blac, 'c.', label='Body Long Accel')
    plt.plot(time, ctac, 'g.', label='Cross Track Accel')
    ax4.set_title('(d) Body Axis Accel', fontsize=20, fontweight="bold")
    plt.ylabel('Acceleration (G)',fontsize=18)
    plt.legend(fontsize=legendfont,loc='upper left');
    plt.xlabel('Time (GMT secs)',fontsize=18);

else:
    #----------------nice plots-----------
    fig = plt.figure(figsize=(30, 15))
    #fig.suptitle('Flight Values Comparison Plot',fontsize=20)

    legendfont = 16


    #number of vertical subplots
    nv = 4

    #------
    #subplot 1
    ax1 = plt.subplot(nv,1,1)
    plt.plot(time, alt,  'ro', label='Pressure Altitude "truth" (ALT) 4Hz')
    #plt.plot(time, baro1,'b.', label='Baro Correct Altitude (BAL1) 4Hz')
    plt.plot(time, ralt, 'g.', label='Radio Altitude (RALT) 8Hz')
    #plt.plot(time, baro2,'g-', label='Baro Correct Altitude (BAL2)')
    #plt.plot(time, alts, 'y.', label='Selected Alt (ALTS)')
    ax1.set_title('(a) Altitude', fontsize=20, fontweight="bold")
    plt.ylabel('Altitude (FT)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='upper left');

    #------
    #subplot 2
    ax2 = plt.subplot(nv,1,2)
    plt.plot(time, altr,'r.', label='Altitude Rate (ALTR) 4Hz')
    plt.plot(time, ivv, 'b.', label='Inertial Vert Speed (IVV) 16Hz')
    plt.plot(time, vsps,'y.', label='Selected Vert Speed (VSPS) 1Hz')
    ax2.set_title('(b) Altitude Rate', fontsize=20, fontweight="bold")
    plt.ylabel('Altitude Rate (FT/Min)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='upper left');

    #------
    #subplot 3
    ax3 = plt.subplot(nv,1,3)
    plt.plot(time, vertg,'r.', label='Vertical Acceleration (VRTG) 8Hz')
    plt.plot(time, latg, 'yo', label='Lateral Acceleration (LATG) 4Hz')
    plt.plot(time, long, 'ko', label='Longitudinal Acceleration (LONG) 4Hz')
    #plt.plot(time, fpac, 'b.', label='Flight Path Acceleration (FPAC)')
    #plt.plot(time, blac, 'c.', label='Body Long Acceleration (BLAC)')
    #plt.plot(time, ctac, 'g.', label='Cross Track Acceleration (CTAC)')
    ax3.set_title('(c) Earth Axis Acceleration', fontsize=20, fontweight="bold")
    plt.ylabel('Acceleration (G)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont,loc='lower left');

    #------
    #subplot 4
    ax4 = plt.subplot(nv,1,4)
    plt.plot(time, fpac, 'b.', label='Flight Path Acceleration (FPAC) 16Hz')
    plt.plot(time, blac, 'c.', label='Body Long Acceleration (BLAC) 16Hz')
    plt.plot(time, ctac, 'g.', label='Cross Track Acceleration (CTAC) 16Hz')
    ax4.set_title('(d) Body Axis Acceleration', fontsize=20, fontweight="bold")
    plt.ylabel('Acceleration (G)',fontsize=18)
    plt.legend(fontsize=legendfont,loc='upper left');
    plt.xlabel('Time (GMT secs)',fontsize=18);

    
#plt.savefig("sensorfig1.pdf", bbox_inches='tight')
#fig.set_size_inches(5,2.5)
#plt.savefig("figures/sensorfig1.pdf", bbox_inches='tight', dpi='figure')

plt.savefig("figures/sensorfig_30x15.png", bbox_inches='tight', dpi='figure')


# In[9]:


#------
#plot 3
fig = plt.figure(figsize=(15, 5))
plt.plot(time, long, 'ko', label='Longitudinal Acceleration (LONG)')
plt.plot(time, blac, 'c.', label='Body Long Acceleration (BLAC)')
ax = plt.gca()
ax.set_title('Redundant Measurements',fontsize=15)
plt.ylabel('G',fontsize=20)
plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);

#------
#plot 3
fig = plt.figure(figsize=(15, 5))
plt.plot(time, latg, 'yo', label='Lateral Acceleration (LATG)')
plt.plot(time, ctac, 'g.', label='Cross Track Acceleration (CTAC)')
ax = plt.gca()
ax.set_title('Acceleration Sensors')
plt.ylabel('G',fontsize=20)
plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);

#------
#plot 3
fig = plt.figure(figsize=(15, 5))
plt.plot(time, vertg,'r.', label='Verticle Acceleration (VRTG)')
plt.plot(time, fpac, 'b.', label='Flight Path Acceleration (FPAC)')
ax = plt.gca()
ax.set_title('Acceleration Sensors')
plt.ylabel('G',fontsize=20)
plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);


# ### Looking at the outlier values in the 3 directional accelerations

# In[10]:


temp = df[['time','VRTG','LATG','LONG']]
i = np.argmin(temp['VRTG'])
j = np.argmin(temp['LATG'])
k = np.argmin(temp['LONG'])


print(temp.iloc[i-5:i+5])
print('\n',temp.iloc[j-5:j+5])
print('\n',temp.iloc[k-5:k+5])


# In[29]:


#---
#true airspeed
tas = df['TAS']
#computed airspeed
cas = df['CAS']
#ground speed
gs = df['GS']
#selected airspeed
cass = df['CASS']
#wind speed
ws = df['WS']

#---
pitch = df['PTCH']
roll = df['ROLL']
da = df['DA'] #drift angle (maybe this is yaw?)


#---
#total air temp
tat = df['TAT']
#static air temp
sat = df['SAT']


plots4paper = True

if plots4paper:
    #----------------smaller plots for the paper-----------

    fig = plt.figure(figsize=(15, 15))
    #fig.suptitle('Flight Values Comparison Plot',fontsize=20)
    #number of vertical subplots
    nv = 4

    #------
    #subplot 1
    ax1 = plt.subplot(nv,1,1)
    plt.plot(time, psa, 'r.', label='Average Static Pressure ')
    plt.plot(time, pi,  'b.', label='Impact Pressure')
    plt.plot(time, pt,  'k.', label='Total Pressure')
    #plt.plot(time, ps,  'c.', label='Static Pressure (PS)') #same as average static pressure
    ax1.set_title('(a) Pressure', fontsize=20, fontweight="bold")
    plt.ylabel('Pressure (mBars)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 2
    ax3 = plt.subplot(nv,1,2)
    plt.plot(time, sat,'r.', label='Static Air Temp')
    plt.plot(time, tat,'k.', label='Total Air Temp')
    plt.ylabel('Temp (Deg)',fontsize=18)
    ax3.set_title('(b) Temperature', fontsize=20, fontweight="bold")
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 3
    ax2 = plt.subplot(nv,1,3)
    plt.plot(time, tas,  'r.', label='True Airspeed')
    plt.plot(time, cas,  'g.', label='Calculated Airspeed')
    plt.plot(time, gs,   'k.', label='Groundspeed')
    plt.plot(time, ws,   'b.', label='Wind Speed')
    plt.plot(time, cass, 'y.', label='Selected Airspeed')
    ax2.set_title('(c) Airspeed', fontsize=20, fontweight="bold")
    plt.ylabel('Speed (Knots)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 4
    ax3 = plt.subplot(nv,1,4)
    plt.plot(time, pitch,'r.', label='Aircraft Pitch')
    plt.plot(time, roll, 'b.', label='Roll Angle')
    plt.plot(time, da,   'g.', label='Drift Angle')
    plt.ylabel('Pitch (Deg)',fontsize=18)
    ax3.set_title('(d) Aircraft Orientation', fontsize=20, fontweight="bold")
    plt.legend(fontsize=legendfont,loc='lower left');
    plt.xlabel('Time (GMT secs)',fontsize=18);

else: 
    #----------------nice plots-----------


    fig = plt.figure(figsize=(30, 15))
    #fig.suptitle('Flight Values Comparison Plot',fontsize=20)
    #number of vertical subplots
    nv = 4

    #------
    #subplot 1
    ax1 = plt.subplot(nv,1,1)
    plt.plot(time, psa, 'ro', label='Average Static Pressure (PSA) 2Hz')
    plt.plot(time, pi,  'b.', label='Impact Pressure (PI) 2Hz')
    plt.plot(time, pt,  'k.', label='Total Pressure (PT) 2Hz')
    #plt.plot(time, ps,  'c.', label='Static Pressure (PS)') #same as average static pressure
    ax1.set_title('(a) Pressure', fontsize=20, fontweight="bold")
    plt.ylabel('Pressure (mBars)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 2
    ax3 = plt.subplot(nv,1,2)
    plt.plot(time, sat,'ro', label='Static Air Temp (SAT) 1Hz')
    plt.plot(time, tat,'k.', label='Total Air Temp (TAT) 1Hz')
    plt.ylabel('Temp (Deg)',fontsize=18)
    ax3.set_title('(b) Temperature', fontsize=20, fontweight="bold")
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 3
    ax2 = plt.subplot(nv,1,3)
    plt.plot(time, tas,  'r+', label='True Airspeed (TAS) 4Hz')
    plt.plot(time, cas,  'g*', label='Calculated Airspeed (CAS) 4Hz')
    plt.plot(time, gs,   'k.', label='Groundspeed (GS) 4Hz')
    plt.plot(time, ws,   'b.', label='Wind Speed (WS) 4Hz')
    plt.plot(time, cass, 'y.', label='Selected Airspeed (CASS) 1Hz')
    ax2.set_title('(c) Airspeed', fontsize=20, fontweight="bold")
    plt.ylabel('Speed (Knots)',fontsize=18)
    plt.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
    plt.legend(fontsize=legendfont);

    #------
    #subplot 4
    ax3 = plt.subplot(nv,1,4)
    plt.plot(time, pitch,'r+', label='Aircraft Pitch (PTCH) 8Hz')
    plt.plot(time, roll, 'b.', label='Roll Angle (ROLL) 8Hz')
    plt.plot(time, da,   'g.', label='Drift Angle (DA) 4Hz')
    plt.ylabel('Pitch (Deg)',fontsize=18)
    ax3.set_title('(d) Aircraft Orientation', fontsize=20, fontweight="bold")
    plt.legend(fontsize=legendfont,loc='lower left');
    plt.xlabel('Time (GMT secs)',fontsize=18);




#plt.savefig("figures/sensorfig2.pdf", bbox_inches='tight', dpi='figure')
plt.savefig("figures/sensorfig2.png", bbox_inches='tight', dpi='figure')
#subplot 5


# In[12]:


#latitude position
lat = df['LATP']
#longitudinal position
lon = df['LONP']

#bounding box
box = ((lon.min(), lon.max(),
        lat.min(), lat.max()))
box = ((-93.483, -83.983, 38.836, 45.236)) #actual bounding box of map image
box


# Go to [openstreetmap.org](https://www.openstreetmap.org/export#map=5/51.500/-0.100) and enter the bounding box coordinates, then take a screen capture
# of the exact square that is outlined.

# In[15]:


mapimg = plt.imread('map.png')

fig1 = plt.figure(figsize=(20, 15))
#fig1.suptitle('Aircraft Position',fontsize=20)

plt.plot(lon, lat,  'b+', label='Lat/Long position(LATP/LONP) 1Hz')
plt.plot(lon[1000], lat[1000],  'ro', label='Starting position')
ax = plt.gca()
#ax.set_title('Aircraft Lat/Long History',fontsize=15)

ax.imshow(mapimg, zorder=0, extent=box, aspect='equal')

plt.xlabel('Longitude (Deg)',fontsize=20)
plt.ylabel('Latitude (Deg)',fontsize=20)
plt.legend(fontsize=legendfont,loc='lower left');
plt.savefig("figures/sensorfig3.pdf", bbox_inches='tight', dpi='figure')
plt.savefig("figures/sensorfig3.png", bbox_inches='tight', dpi='figure')
#minneapolis to cincinnati?


# In[145]:


#elevator positions
elevl = df['ELEV_1']
elevr = df['ELEV_2']

fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Aircraft Control Surfaces',fontsize=20)

plt.plot(time, elevl,  'b+', label='Left Elevator (ELEV_1)')
plt.plot(time, elevr,  'k+', label='Right Elevator (ELEV_2)')

plt.ylabel('Control Surface Position (Deg)',fontsize=20)
plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);


##getting rid of inner labels
#for ax in fig.get_axes():
#    ax.label_outer()


# In[146]:


#elevator positions
nsqt = df['NSQT']
#elevr = df['ELEV_2']

fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Airborne status compared to Ground Speed & alt',fontsize=20)


#plt.plot(time, vertg,'r.', label='Verticle Acceleration (VRTG) 8Hz')
plt.plot(time, nsqt,  'b+', label='front landing gear switch (NSQT) 4Hz')
plt.plot(time, gs,  'k.', label='Ground Speed (GS) 4Hz')
plt.plot(time, alt,  'g.', label='Alt')
#plt.plot(time, elevr,  'k+', label='Right Elevator (ELEV_2)')

plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);
plt.xlim(74300,74550)
plt.ylim(0,200)

##getting rid of inner labels
#for ax in fig.get_axes():
#    ax.label_outer()


# In[147]:


#elevator positions
nsqt = df['NSQT']
#elevr = df['ELEV_2']
ig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Airborne status compared to Ground Speed & alt',fontsize=20)
fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Ground Speed',fontsize=20)

plt.plot(time, gs,  'k.', label='Ground Speed (GS) 4Hz')
#plt.plot(time, elevr,  'k+', label='Right Elevator (ELEV_2)')

plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);


##getting rid of inner labels
#for ax in fig.get_axes():
#    ax.label_outer()


# In[90]:


max(gs[0:500])


# In[44]:


#elevator positions
nsqt = df['NSQT']
#elevr = df['ELEV_2']

fig2 = plt.figure(figsize=(20, 15))
fig2.suptitle('Airborne status compared to acceleration noise',fontsize=20)


plt.plot(time, vertg,'r.', label='Verticle Acceleration (VRTG) 8Hz')
plt.plot(time, nsqt,  'b+', label='front landing gear switchNSQT) 4Hz')
#plt.plot(time, elevr,  'k+', label='Right Elevator (ELEV_2)')

plt.xlabel('Time (GMT secs)',fontsize=20)
plt.legend(fontsize=18);
plt.xlim(47100,47110)

##getting rid of inner labels
#for ax in fig.get_axes():
#    ax.label_outer()


# In[122]:


ns_qt = np.nan
ground_speed = 0

airborne = ns_qt or ground_speed

if not ns_qt or not ground_speed:
    print('ground')

airborne


# In[152]:


df_drop = df.index[not df['GS'][:].bool()]


# In[200]:


get_ipython().run_cell_magic('time', '', "\nn_samples = gs.shape[0]\nprint(f'n_samples: {n_samples}')\n\n\ndrop_list = []\ni=0\nwhile i < n_samples-4:\n    #print(i)\n    gndspd = gs[i]\n    if not gndspd:\n        #print(f'i = {i}')\n        drop_list.append(i)\n        drop_list.append(i+1)\n        drop_list.append(i+2)\n        drop_list.append(i+3)\n        i += 4\n    else:\n        i += 1\n    \nlen(drop_list), drop_list\n")


# In[190]:


get_ipython().run_line_magic('timeit', '-n 10 df.iloc[df.index.drop(drop_list)]')
get_ipython().run_line_magic('timeit', '-n 10 df.loc[df.index.drop(drop_list)]')
get_ipython().run_line_magic('timeit', '-n 10 df[~df.index.isin(drop_list)]')
get_ipython().run_line_magic('timeit', '-n 10 df.drop(drop_list)')


# In[201]:


get_ipython().run_line_magic('timeit', "-n 10 [i for i in range(n_samples) if not df['GS'][i]]")


# In[209]:


get_ipython().run_cell_magic('time', '', 'df = df.iloc[df.index.drop(drop_list)]\ndf\n')


# In[208]:


df.shape[0]


# In[72]:


i=0
while i < n_samples-8:
    print(i)
    if i == 5:
        print(f'i = {i}')
        i += 2
    i+=1


# In[ ]:




