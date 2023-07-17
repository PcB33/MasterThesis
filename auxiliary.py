import numpy as np
import lifesim as ls
import requests


#some conversions and constants
au2par = 4.8481368111358*10**-6
par2au = 1/au2par
rad2arcsec = 180/np.pi*3600
arcsec2rad = 1/rad2arcsec
m_per_pc = 3.086e+16
m_per_au = 1.49597870691e+11

h = 6.62607e-34
k = 1.380649e-23
c = 2.99792e+8

R_sun  = 6.947e+8
T_sun = 5778.

R_earth = 6.371e+6
T_earth = 276.

def convert_to_minus_pi_to_pi(angle):
    return (((angle + np.pi) % (2*np.pi)) - np.pi)


def convert_to_zero_to_2pi(angle):
    if (angle<0):
        angle += 2*np.pi
    return angle

#path to master thesis folder
path='C:/Users/Philipp Binkert/OneDrive/ETH/Master_Thesis/'

#start GUI
def start_Gui():
    ls.Gui()
    return

#start_Gui()

#download ppop-file (only once)
def download_PPOP():
    data = requests.get('https://raw.githubusercontent.com/kammerje/P-pop/main/TestPlanetPopulation.txt')
    with open(path+'04_input_files/ppop_catalog_10.txt', 'wb') as file:
        file.write(data.content)

    return