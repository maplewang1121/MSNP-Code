import numpy as np
import matplotlib.pyplot as plt
import scipy

data_file = input('Enter Data File:\n')
set_name = input('Enter a name for this dataset:\n')
set_data = np.sort(np.loadtxt(data_file,skiprows=1,unpack=True,encoding='UTF-8'))
# Setting up the graph axes



OFFSET = 5
BINS = [30,35,40,45,50,55,60,65,70,75]
MIN_RANGE = np.round(np.min(set_data),2) - OFFSET
MAX_RANGE = np.round(np.max(set_data),2) + OFFSET
data_labels = ['d11','d12','d13','d14','d22','d23','d24']
colors = ['red','green','blue','yellow','purple','brown','orange']
#plt.title("Counts of MSNP sizes by Diameter (nm) of " + set_data_str + ' Data',)

plt.xlabel("Diameter (nm)")
plt.xlim(min(BINS),max(BINS))
plt.ylabel("Counts")
plt.rcParams.update({'font.size': 16})
plt.title("MSNP sizes by Diameter (nm) of " + set_name)

def get_frequency():
    new_list = []
    total = len(set_data)
    i = 0
    while i<len(BINS)-1:
        tally = 0
        for j in set_data:
            if (BINS[i]<j and BINS[i+1]>j):
                tally += 1
        total += tally
        new_list.append(tally/total)
        i += 1
    return new_list
        
## Plotting the actual data


## Fitting a normal distribution
mu, std = scipy.stats.norm.fit(set_data)

def pdf(x,mu,std):
    return (1/(std*(2*np.pi)**0.5)*np.exp((-0.5*((x-mu)/std)**2)))

pdf = pdf(set_data,mu,std)


def get_scale():
    return y_range[1]/np.max(pdf)

CV = std/mu

def plot_data():
    return plt.hist(set_data,bins=BINS,rwidth=0.8,color='red')

plot_data()

y_range = plt.gca().get_ylim()
pdf_scale = pdf * get_scale()

def plot_trendline():
    plt.plot(set_data,pdf_scale,linewidth=1,color='blue',marker='',label='$d_{TEM}$ = '\
                + str(round(mu,3)) + '\nstd = ' + str(round(std,3)) + '\nCV = '\
                    + str(round(CV,3)) + '\nn = ' + str(len(set_data)))
    return


plot_trendline()

"""def change_set(new_set):
    if new_set in data_dict:
        set_data = np.sort(data_dict[new_set])
        mu, std = scipy.stats.norm.fit(set_data)
        pdf = pdf(set_data,mu,std)
        
    else:
        print('Invalid Set')
        return"""


fig = plt.gcf()
plt.rcParams.update({'font.size': 12})
plt.legend(loc='upper right')

def save_plot():
    filename = input('Enter File Name and Type:\n')
    fig.savefig(filename, dpi=900)
    return 'Saved as ' + filename