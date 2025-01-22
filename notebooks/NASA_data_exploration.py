#!/usr/bin/env python
# coding: utf-8

# $$\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\Yv}{\mathbf{Y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\betav}{\mathbf{\beta}}
# \newcommand{\gv}{\mathbf{g}}
# \newcommand{\Hv}{\mathbf{H}}
# \newcommand{\dv}{\mathbf{d}}
# \newcommand{\Vv}{\mathbf{V}}
# \newcommand{\vv}{\mathbf{v}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\Sv}{\mathbf{S}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\Zv}{\mathbf{Z}}
# \newcommand{\Norm}{\mathcal{N}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# \newcommand{\dimensionbar}[1]{\underset{#1}{\operatorname{|}}}
# \newcommand{\grad}{\mathbf{\nabla}}
# \newcommand{\ebx}[1]{e^{\wv_{#1}^T \xv_n}}
# \newcommand{\eby}[1]{e^{y_{n,#1}}}
# \newcommand{\Tiv}{\mathbf{Ti}}
# \newcommand{\Fv}{\mathbf{F}}
# \newcommand{\ones}[1]{\mathbf{1}_{#1}}
# $$

# ## UNCLASSIFIED 
# _Distribution A_

# # Data exploration and visualization for NASA data
# data gathered from: https://c3.ndc.nasa.gov/dashlink/projects/85/resources/

# The purpose of this file is to figure out the format of the matlab files with the intention of converting them
# to panda dataframes and saving them in a parquet format.  all files that were downloaded are in matlab (.mat) format.

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import dateutil
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')


# ---

# ## Parameters Contained in each file

# In[8]:


data1 = loadmat('Tail_687_1/687200104111158.mat')
data1 = loadmat('Tail_654_4/654200204300312.mat')
k = data1.keys()
k


# In[9]:


len(k)


# ## All dictionary values are made of an numpy ndarray of objects
# ### They all have the objects named 'data', 'Rate', 'Units', 'Description', 'Alpha'
# #### With the exception of the '__header__', '__version__', '__globals__' values

# In[10]:


[print(data1[k].dtype.names) for k in data1.keys() if type(data1[k]) == np.ndarray]


# In[11]:


header = data1['__header__']
version = data1['__version__']
gs = data1['__globals__']
print(f'header = :{header}')
print(f'version = :{version}')
print(f'globals = :{gs}')


# # Reading and formatting Metadata for the file

# ## Calculating metadata

# In[12]:


import os

fdir = 'Tail_687_1'

file = '687200104111158'
metadata = []
metadata = defaultdict(dict)

#filelist = os.listdir(fdir)
filelist = [(f'{file}.mat')]
#print(len(filelist))
#for f in filelist[0:12]:
for f in filelist:
    #print(f)
    d = loadmat((f'{fdir}/{f}'))
    year   = d['DATE_YEAR'][0][0][0][:]
    month  = d['DATE_MONTH'][0][0][0][:]
    day    = d['DATE_DAY'][0][0][0][:]
    hour   = d['GMT_HOUR'][0][0][0][:]
    minute = d['GMT_MINUTE'][0][0][0][:]
    second = d['GMT_SEC'][0][0][0][:]
    ys = min(np.unique(year))
    ms = min(np.unique(month))
    ds = min(np.unique(day))
    
    #h0s = hour[0][0].astype(float)
    #m0s = minute[0][0].astype(float)
    #s0s = second[0][0].astype(float)
    #h0e = hour[-1][0].astype(float)
    #m0e = minute[-1][0].astype(float)
    #s0e = second[-1][0].astype(float)
    ##starttime = timedelta(hours=h0s, minutes=m0s, seconds=s0s)
    ##stoptime = timedelta(hours=h0e, minutes=m0e, seconds=s0e)
    ##print(f'First file:{year[0,0]}:{month[0,0]}:{day[0,0]} Start: {starttime}, Stop {stoptime}')
    #print(f'File {f}:   {ys}:{month[0,0]}:{day[0,0]} Start: {h0s}:{m0s}:{s0s}, Stop {h0e}:{m0e}:{s0e}')
    if ys == 2165: 
        airbornedata = False
    else:
        airbornedata = True
        
    metadata[file]['AirborneData'] = airbornedata
    
    metadata[file]['StartDate'] = {}
    metadata[file]['StartDate']['Year']  = ys.astype(str)
    metadata[file]['StartDate']['Month'] = ms.astype(str)
    metadata[file]['StartDate']['Day']   = ds.astype(str)
    
        
    


# ### Stored Metadata

# In[13]:


from collections import defaultdict

for k in data1.keys():
    #don't print the first "__xxxx__" parameters
    if type(data1[k]) == np.ndarray:
        u = data1[k]["Units"][0,0] 
        if u.shape[0] == 0: u = ["<none>"]
        #metadata[file] = defaultdict(dict)
        #metadata[file][k] = defaultdict(dict)
        metadata[file][k] = {}
        metadata[file][k]['Units']       = u[0]
        metadata[file][k]['Rate']        = data1[k]["Rate"][0,0][0,0].astype('float')
        metadata[file][k]['Alpha']       = data1[k]["Alpha"][0,0][0]
        metadata[file][k]['Description'] = data1[k]["Description"][0,0][0]
        #check to see if the "units" are blank and set to '<none>'
        print(f'{k:12s}Units: {str(metadata[file][k]["Units"]):10s}Rate: {str(metadata[file][k]["Rate"]):5s}',
                f'Alpha: {metadata[file][k]["Alpha"]:11s} Description: {metadata[file][k]["Description"]:10s}')
        


# In[14]:


len(metadata[file])


# ### Writing metadata to a JSON file

# In[18]:


import json
with open((f'{file}.json'), 'w') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)


# ## Function

# In[19]:


def mat_2_df(mdata, param, samplesize):
    '''This NASA matlab data is a dictionary of arrays of objects and this function
    takes in a single dictionary word and parses the data into a pandas dataframe
    '''
    #create an array filled with NaNs so we don't fill in data with bad data yet.
    d = np.empty(samplesize).reshape(-1,1)
    d[:] = np.NaN
    rate = mdata[param]['Rate'][0,0][0,0]
    i = int(16/rate)
    d[::i] = mdata[param]['data'][0,0].reshape(-1,1)

    return d
    


# ## Figuring out the data rates compared to the GMT hour:minute:seconds

# In[20]:


hour   = data1['GMT_HOUR']['data'][0,0].reshape(-1,1).astype(int)
minute = data1['GMT_MINUTE']['data'][0,0].reshape(-1,1).astype(int)
second = data1['GMT_SEC']['data'][0,0].reshape(-1,1).astype(int)

#set t array as large as 16hz signal as NaNs then add actual time in GMT
GMT = np.array(hour*3600 + minute*60 + second)

clockstart = (f'{hour[0,0]}:{minute[0,0]}:{second[0,0]}')
clockstop  = (f'{hour[-1,0]}:{minute[-1,0]}:{second[-1,0]}')
clocktime  = [(f'{hour[i,0]}:{minute[i,0]}:{second[i,0]}') for i in range(hour.shape[0])]

starttime = GMT[1,0] - 0.5
stoptime  = GMT[-1,0] + 4*0.5
diff = stoptime - starttime

aoa1_rate = data1['AOA1']['Rate'][0,0][0,0]
aoa1_data_size = data1['AOA1']['data'][0][0].shape[0] 

gmt_sec_rate = data1['GMT_SEC']['Rate'][0,0][0,0]

timesamples = hour.shape[0]

print(f'Starting time of file: {clockstart}, {starttime} seconds')
print(f'Ending time of file: {clockstop}, {stoptime} seconds')
print(f'AOA1 rate: {aoa1_rate}')
print(f'GMT_sec rate: {gmt_sec_rate}')
print(f'diff {diff}')
print(f'AOA1 sample size {aoa1_data_size:,}')
print(f'time sample size {timesamples:,}')
print(f'AOA1 calculated rate (samples/sec) {aoa1_data_size/diff} Hz')
print(f'rate (AOA1 samples/time samples) {aoa1_data_size/(timesamples)}')


# ## Creating the full dataframe

# In[22]:


hour   = data1['GMT_HOUR']['data'][0,0].reshape(-1,1).astype(int)
minute = data1['GMT_MINUTE']['data'][0,0].reshape(-1,1).astype(int)
second = data1['GMT_SEC']['data'][0,0].reshape(-1,1).astype(int)

#set t array as large as 16hz signal as NaNs then add actual time in GMT
GMT = np.array(hour*3600 + minute*60 + second)

#matches the fastest 16 Hz (adding an extra amount above 16Hz found in the data
t_delta = 0.0625 + 0.5/549/16#seconds. 

samplesize_16hz = data1['FPAC']['data'][0,0].shape[0]

#the first GMTsecs
t0 = GMT[0]

#find the first time GMTsecs transitions to a new time, this is reliable GMT
#it normally stays the same time for 6 seconds.
i = min([i for i in range(12) if not t0 == GMT[i]])

#GMTsecs is 2 Hz or 0.5 seconds
timeoffset = 0.5*i

#offset from the first reliable GMT
starttime = GMT[i,0] - timeoffset 

stoptime = starttime + t_delta*(samplesize_16hz-1) 

#create a 16 Hz time column to line everything up and make it a pandas dataframe
t16hz = np.linspace(starttime, stoptime, samplesize_16hz)
df1 = pd.DataFrame(data={'time':t16hz})

#adding GMT in seconds
t = np.empty(samplesize_16hz).reshape(-1,1)
t[:] = np.NaN
t[::8] = GMT #every eighth element
df1['GMTsecs'] = t

#looping through all of the keys and creating a full Pandas DataFrame
for k in data1.keys():
    #don't print the first "__xxxx__" parameters
    if type(data1[k]) == np.ndarray:
       
        df1[k] = mat_2_df(data1, k, samplesize_16hz)


#removing the first set of unreliable GMT data so it starts cleanly
df1 = df1[i*8:-1]
df1.reset_index(inplace=True)
df1.pop('index')

#writing to a parquet file
df1.to_parquet(path=(f'{file}.parquet'), compression='gzip')

df1


# ## Inconsistent GMT seconds
# 
# 
# ### some of these GMT samples only have 11 samples instead of 12 like they should.  not sure why?
# For instance 43410 GMT is the first one to have 11 samples instead of 12.
# This made 43415.5 16hz time match up with 43416 GMT<br>
# If all other signals line up with the GMT secs then I should modify the 8 NaNs on all other signals to make the 
# GMT signal (43416) line up with the 16hz 43416 time stamp
# 
# ## solved by adding 0.5/549/16 seconds to each 16Hz signal.  
# As seen below, if you round the 16Hz signal to nearest int when comparing GMTsecs and 16Hz they will align 
# properly
# 
# 

# In[1]:


1/16


# In[364]:


#checking to ensure the times are lining up.
df1 = pd.read_parquet(f'{file}.parquet')
istart = 0
istop = -1
newsamplesize_16hz = df1.shape[0]

#find the first non-NaN number in GMTsecs
t0 = [df1["GMTsecs"][i] for i in range(istart,istart+8) if not np.isnan(df1["GMTsecs"][i])][0]

mismatch = False

for i in range(newsamplesize_16hz):
    gmtsecs = df1["GMTsecs"][i]
    hz16 = df1["time"][i]
    hzorig = hz16 
    
    #compare to previous GMTsecs to see when it changes and if it matches up
    # with the 16Hz signal time when it does change
    if not np.isnan(gmtsecs) and not t0 == gmtsecs:
        #compare integer values
        hz16 = round(hz16,0)
        gmtsecs = round(gmtsecs,0)
        t0 = gmtsecs
        if not gmtsecs == hz16:
            print(f'gmtsecs={gmtsecs}, 16Hz = {hz16}, 16Hzorig={round(hzorig,0)}')
            mismatch = True
        
if not mismatch: print('No mismatches!')


# In[194]:


nanline = pd.DataFrame({"VAR_1107": np.nan},index=[0])
nanline
nanline["VAR_2670"] = 3
nanline.index = [1]
nanline


# In[196]:


nanline = [np.nan for k in data1.keys() if type(data1[k] == np.ndarray)]
#for k in data1.keys():
#    #don't print the first "__xxxx__" parameters
#    if type(data1[k]) == np.ndarray:
#       
#        nanline[k] = np.nan
nanline


# In[161]:


istart = 5660
#t0 = [df1["GMTsecs"][i] for i in range(istart,istart+8) if np.isreal(df1["GMTsecs"][i]) ]
t0 = [df1["GMTsecs"][i] for i in range(istart,istart+8) if not np.isnan(df1["GMTsecs"][i])][0]

t0


# In[151]:


np.isnan(t0[0])


# ## Each file represents a single flight 

# checking to see if each successive file is ordered in time.
#   It is, but I found a few that had mislabeled names for their time.

# In[33]:


from datetime import timedelta
data0 = loadmat('Tail_687_1/687200103200323.mat')
data01 = loadmat('Tail_687_1/687200103200350.mat')


# In[41]:


#first file in Tail 687_1
hour   = data0['GMT_HOUR'][0][0][0][:]
minute = data0['GMT_MINUTE'][0][0][0][:]
second = data0['GMT_SEC'][0][0][0][:]
year   = data0['DATE_YEAR'][0][0][0][:]
month  = data0['DATE_MONTH'][0][0][0][:]
day    = data0['DATE_DAY'][0][0][0][:]
h0s = hour[0][0].astype(float)
m0s = minute[0][0].astype(float)
s0s = second[0][0].astype(float)
h0e = hour[-1][0].astype(float)
m0e = minute[-1][0].astype(float)
s0e = second[-1][0].astype(float)
starttime = timedelta(hours=h0s, minutes=m0s, seconds=s0s)
stoptime = timedelta(hours=h0e, minutes=m0e, seconds=s0e)
#print(f'First file:{year[0,0]}:{month[0,0]}:{day[0,0]} Start: {starttime}, Stop {stoptime}')
print(f'First file:{year[0,0]}:{month[0,0]}:{day[0,0]} Start: {h0s}:{m0s}:{s0s}, Stop {h0e}:{m0e}:{s0e}')

#second file in Tail 687_1
hour   = data01['GMT_HOUR'][0][0][0][:]
minute = data01['GMT_MINUTE'][0][0][0][:]
second = data01['GMT_SEC'][0][0][0][:]
year   = data01['DATE_YEAR'][0][0][0][:]
month  = data01['DATE_MONTH'][0][0][0][:]
day    = data01['DATE_DAY'][0][0][0][:]
h01s = hour[0][0].astype(float)
m01s = minute[0][0].astype(float)
s01s = second[0][0].astype(float)
h01e = hour[-1][0].astype(float)
m01e = minute[-1][0].astype(float)
s01e = second[-1][0].astype(float)
starttime = timedelta(hours=h01s, minutes=m01s, seconds=s01s)
stoptime = timedelta(hours=h01e, minutes=m01e, seconds=s01e)

print(f'Second file:{year[0,0]}:{month[0,0]}:{day[0,0]} Start: {h01s}:{m01s}:{s01s}, Stop {h01e}:{m01e}:{s01e}')
#print(f'Second file:{year[0,0]}:{month[0,0]}:{day[0,0]} Start: {starttime}, Stop {stoptime}')




# In[42]:


np.unique(year)


# In[54]:


df2 = pd.read_parquet(path='Tail_687_1_687200104111158.parquet')
df2


# ## Plotting elevation values 

# In[1]:


alt = data1['ALT']['data'][0,0].reshape(1,-1)[0]
altr = data1['ALTR']['data'][0,0].reshape(1,-1)[0]
ralt = data1['RALT']['data'][0,0].reshape(1,-1)[0]
#calt = data1['CALT']['data'][0,0].reshape(1,-1)[0]

nsamples = data1['ALT']['data'][0,0].shape[0]
rsamples = data1['RALT']['data'][0,0].shape[0]

xs = np.linspace(1,nsamples,nsamples)

xr = np.linspace(1,nsamples,rsamples)

#alt = alt.astype(np.int)
#alt, x


#alt = data1['ALT']['data'][0,0]

fig1 = plt.figure(figsize=(20, 15))


plt.plot(xs, alt, 'r-', label='alt')
plt.plot(xs, altr, 'b-', label='altr')
plt.plot(xr, ralt, 'g-', label='altr')
#ax=plt.subplot(aspect='equal')



plt.ylabel('y')
plt.xlabel('x')
ax.set_title('Elevation')
plt.label()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


#plt.xlim([xmin, xmax])
#plt.ylim([ymin, ymax])


# ---

# In[53]:


ralt = data.get('RALT')
print(f'RALT min/max: {min(ralt[0][0][0])[0]}/{max(ralt[0][0][0])[0]:,}')
print(f'Number of RALT samples: {ralt[0][0][0].shape[0]:,}')

calt = data.get('CALT')
print(f'CALT min/max: {min(calt[0][0][0])[0]}/{max(calt[0][0][0])[0]:,}')
print(f'Number of CALT samples: {calt[0][0][0].shape[0]:,}')

altr = data.get('ALTR')
print(f'ALTR min/max: {min(altr[0][0][0])[0]:,}/{max(altr[0][0][0])[0]:,}')
print(f'Number of ALTR samples: {altr[0][0][0].shape[0]:,}')

alts = data.get('ALTS')
print(f'ALTS min/max: {min(alts[0][0][0])[0]:,}/{max(alts[0][0][0])[0]:,}')
print(f'Number of ALTS samples: {alts[0][0][0].shape[0]:,}')

alt = data.get('ALT')
print(f'ALT min/max: {min(alt[0][0][0])[0]}/{max(alt[0][0][0])[0]:,}')
print(f'Number of ALT samples: {alt[0][0][0].shape[0]:,}')


# In[538]:


print(f'samples of MACH: {np.size(data1.get("MACH")[0][0][0]):,}')
print(f'rate of MACH: {np.size(data1["MACH"]["Rate"][0, 0][0])}')


# ### APFD is Auto Pilot Flight Director Status and is "probably" 0 when it's on the ground

# In[47]:


np.unique(data1['APFD']['data'][0,0][:,0])


# #### Velocity shows that it's on the ground

# In[44]:


max(data1['MACH']['data'][0,0][:,0])


# ---

# ---

# In[52]:


data = loadmat('Tail_687_1/687200109140804.mat')
type(data)


# ### APFD shows that this is not on the ground the whole time and has a bit of all states in this file

# In[48]:


np.unique(data.get('APFD')[0][0][0])


# #### Velocity shows that it's NOT on the ground

# In[47]:


max(data.get('MACH')[0][0][0])


# Listing all of the available keys in the dictionary

# In[49]:


data.keys()


# #### Not sure what some of these values are so i'm printing them here

# ## Disecting a single name in the dictionary that includes data and metadata

# In[331]:


mdata = data['AOA1']
mdtype = mdata.dtype #objects in mat file

print(f'Data names in AOA1: {mdtype.names}')

ndata = {n: mdata[n][0, 0] for n in mdtype.names}


#[print(f'A0A1 {r}: {ndata[r]}') for r in mdtype.names]

aoa_data =        ndata['data'].reshape(1,-1)[0]
aoa_rate =        ndata["Rate"][0][0]
aoa_units =       ndata["Units"][0]
aoa_description = ndata["Description"][0]
aoa_alpha =       ndata["Alpha"][0]

aoa_samples = data["AOA1"][0][0][0].shape[0]

print(f'Number of AOA1 samples in dataset = {aoa_samples:,}')
print(f'A0A1 data: {aoa_data}')
print(f'A0A1 Rate: {aoa_rate}')
print(f'A0A1 Units: {aoa_units}')
print(f'A0A1 Description: {aoa_description}')
print(f'A0A1 Alpha: {aoa_alpha}')


d = {'AOA1': aoa_data}
df = pd.DataFrame(data=d)
print(f'\nPandas DataFrame:\n{df}');
#dft = pd.DataFrame.from_dict(ndata)


# In[ ]:


data['RALT']['Units'][0, 0][0]


# In[316]:


52992/2


# ####  Angles of Attack

# In[ ]:


aoa1 = data.get('AOA1')[0][0][0][:]
print(f'AOA1 min/max: {min(aoa1)[0]:.2f}/{max(aoa1)[0]:.2f}')
aoa2 = data.get('AOA2')[0][0][0][:]
print(f'AOA2 min/max: {min(aoa2)[0]:.2f}/{max(aoa2)[0]:.2f}')
pitch = data.get('PTCH')[0][0][0][:]
print(f'PTCH min/max: {min(pitch)[0]:.2f}/{max(pitch)[0]:.2f}')
aoaI = data.get('AOAI')[0][0][0][:]
print(f'AOAI min/max: {min(aoaI)[0]:.2f}/{max(aoaI)[0]:.2f}')
aoaC = data.get('AOAC')[0][0][0][:]
print(f'AOAC min/max: {min(aoaC)[0]:.2f}/{max(aoaC)[0]:.2f}')


# In[ ]:


roll = data.get('ROLL')[0][0][0][:]
print(f'ROLL min/max: {min(roll)[0]:.2f}/{max(roll)[0]:.2f}')


# #### Other Data contained within the Matlab dictionaries
# I'm not sure what this extra data means, but the first 2 array indicies of each of these dictionary values are
# empty.  This may be a good chance to rewrite them as panda parquet files.
# 

# In[79]:


otherdata1 = data.get('AOA1')[0][0][1][:]
otherdata2 = data.get('AOA2')[0][0][1][:]
otherdata3 = data.get('PTCH')[0][0][1][:]
otherdata4 = data.get('AOAI')[0][0][1][:]
otherdata5 = data.get('AOAC')[0][0][1][:]
print(f'otherdata1={otherdata1}')
print(f'otherdata2={otherdata2}')
print(f'otherdata3={otherdata3}')
print(f'otherdata4={otherdata4}')
print(f'otherdata5={otherdata5}')


# ## Comparing different Altitude measurements

# In[53]:


ralt = data.get('RALT')
print(f'RALT min/max: {min(ralt[0][0][0])[0]}/{max(ralt[0][0][0])[0]:,}')
print(f'Number of RALT samples: {ralt[0][0][0].shape[0]:,}')

calt = data.get('CALT')
print(f'CALT min/max: {min(calt[0][0][0])[0]}/{max(calt[0][0][0])[0]:,}')
print(f'Number of CALT samples: {calt[0][0][0].shape[0]:,}')

altr = data.get('ALTR')
print(f'ALTR min/max: {min(altr[0][0][0])[0]:,}/{max(altr[0][0][0])[0]:,}')
print(f'Number of ALTR samples: {altr[0][0][0].shape[0]:,}')

alts = data.get('ALTS')
print(f'ALTS min/max: {min(alts[0][0][0])[0]:,}/{max(alts[0][0][0])[0]:,}')
print(f'Number of ALTS samples: {alts[0][0][0].shape[0]:,}')

alt = data.get('ALT')
print(f'ALT min/max: {min(alt[0][0][0])[0]}/{max(alt[0][0][0])[0]:,}')
print(f'Number of ALT samples: {alt[0][0][0].shape[0]:,}')


# The Day, Month, Year look reasonable.

# In[259]:


year  = data.get('DATE_YEAR')[0][0][0][:]
month = data.get('DATE_MONTH')[0][0][0][:]
day   = data.get('DATE_DAY')[0][0][0][:]

print(f'Unique values of Date_Year:  {np.unique(year)[0]}')
print(f'Unique values of Date_Month: {np.unique(month)[0]}')
print(f'Unique values of Date_Day:   {np.unique(day)[0]}')


# #### It looks like GMT time does not change in this data

# So far this GMT time does not make sense to me.

# In[245]:


hour   = data.get('GMT_HOUR')[0][0][0][:]
minute = data.get('GMT_MINUTE')[0][0][0][:]
second = data.get('GMT_SEC')[0][0][0][:]

#figuring out what the "rate" means
h1 = hour[0][0]
m1 = minute[0][0]
s1 = second[0][0]

hour[-1][0]

#print(f'Unique values of GMT Hour:   {np.unique(hour)}')
#print(f'Unique values of GMT Minute: {np.unique(minute)}')
#print(f'Unique values of GMT Second: {np.unique(second)}')


# ---

# In[109]:


from datetime import timedelta, date

d645 = loadmat('Tail_687_1/687200109140645.mat')
hour   = d645['GMT_HOUR'][0][0][0][:]
minute = d645['GMT_MINUTE'][0][0][0][:]
second = d645['GMT_SEC'][0][0][0][:]
year   = d645['DATE_YEAR'][0][0][0][:]
month  = d645['DATE_MONTH'][0][0][0][:]
day    = d645['DATE_DAY'][0][0][0][:]

#figuring out what the "rate" means
h1 = hour[0][0].astype(float)
m1 = minute[0][0].astype(float)
s1 = second[0][0].astype(float)

starttime = timedelta(hours=h1, minutes=m1, seconds=s1)

he = hour[-1][0].astype(float)
me = minute[-1][0].astype(float)
se = second[-1][0].astype(float)

stoptime = timedelta(hours=he, minutes=me, seconds=se)

timesamples = second.shape[0]


diff = stoptime - starttime 

aoa1_rate = d645['AOA1']['Rate'][0][0][0][0]
aoa1_data_size = d645['AOA1']['data'][0][0].shape[0]




print(f'Date:  {np.unique(year)[0]}')
print(f'Starting time of file: {starttime}')
print(f'Ending time of file: {stoptime}')
print(f'AOA1 rate: {aoa1_rate}')
print(f'AOA1 calc rate (samples/sec) {aoa1_data_size/diff.total_seconds()}')
print(f'diff (secs) {diff.total_seconds():,}')
print(f'AOA1 sample size {aoa1_data_size:,}')
print(f'time sample size {timesamples:,}')
print(f'rate ratio (AOA1 samples/time samples) {aoa1_data_size/(timesamples)}')


date(np.unique(year)[0], np.unique(month)[0], np.unique(day)[0])


# In[112]:


stoptime.seconds - starttime.seconds


# In[116]:


starttime.total_seconds()
#starttime.seconds


# ---

# In[130]:


data350 = loadmat('Tail_687_1/687200103200350.mat')
print(f'type of data350: {type(data350)}\n')
print(f'keys of data350: {(data350.keys())}\n')


# #### Not sure what some of these values are so i'm printing them here

# In[109]:


header = data350.get('__header__')
version = data350.get('__version__')
gs = data350.get('__globals__')
print(f'header = :{header}')
print(f'version = :{version}')
print(f'globals = :{gs}')


# #### Different Angles of Attack

# In[110]:


aoa1 = np.unique(data350.get('AOA1')[0][0][0][:])
print(f'Unique AOA1: {aoa1}')
aoa2 = np.unique(data350.get('AOA2')[0][0][0][:])
print(f'Unique AOA2: {aoa1}')
pitch = np.unique(data350.get('PTCH')[0][0][0][:])
print(f'Unique PTCH: {aoa1}')
aoaI = np.unique(data350.get('AOAI')[0][0][0][:])
print(f'Unique AOAI: {aoaI}')
aoaC = np.unique(data350.get('AOAC')[0][0][0][:])
print(f'Unique AOAC: {aoaC}')


# ### These two files appear to be "created" only 3 seconds apart
# However, I can't seem to find a good timestamp in this dictionary yet to determine if this is matlab timestamp
# or a recording timestamp

# The year, month and days do not make sense

# In[111]:


year  = data350.get('DATE_YEAR')[0][0][0][:]
month = data350.get('DATE_MONTH')[0][0][0][:]
day   = data350.get('DATE_DAY')[0][0][0][:]

print(f'Unique values of Date_Year:  {np.unique(year)}')
print(f'Unique values of Date_Month: {np.unique(month)}')
print(f'Unique values of Date_Day:   {np.unique(day)}')


# #### It looks like GMT time does not change in this data when the plane is on the ground

# GMT time is set to these values when plane is on the ground

# In[112]:


hour   = data350.get('GMT_HOUR')[0][0][0][:]
minute = data350.get('GMT_MINUTE')[0][0][0][:]
second = data350.get('GMT_SEC')[0][0][0][:]

print(f'Unique values of GMT Hour:   {np.unique(hour)[0]}')
print(f'Unique values of GMT Minute: {np.unique(minute)[0]}')
print(f'Unique values of GMT Second: {np.unique(second)[0]}')


# Trying to figure out if this is EGI time

# In[201]:


egt1   = data350['EGT_1'][0][0][0][:]
egt2   = data350['EGT_2'][0][0][0][:]
egt3   = data350['EGT_3'][0][0][0][:]
egt4   = data350['EGT_4'][0][0][0][:]

print(f'Unique values of EGT1: {np.size(egt1)}')
#print(f'Unique values of EGT2: {np.unique(egt2)}')
#print(f'Unique values of EGT3: {np.unique(egt3)}')
#print(f'Unique values of EGT4: {np.unique(egt4)}')


# In[ ]:




