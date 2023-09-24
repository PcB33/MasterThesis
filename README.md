# MasterThesis (from Chapter 2 from the written thesis)
The relevant code consists of the following five files. Let again be noted that large parts of Extraction.py as well as some functions in Extraction_auxiliary.py were taken over or adapted from
the code available for the old LIFEsim version with credit to Dannert et al. (2022) and M. Ottiger
(2020).
• Extraction.py: This is the key file where most of the calculations described in the sections
above occur. Moving forward, this will be referred to as the main file.
• Extraction_auxiliary.py: This file contains several auxiliary functions (such as plots, coordinate transformations etc.) that are imported and called in the main file. They are saved
separately for readability of the main file.
SinglePlanet_extraction.py: This file is used to perform signal extraction for one specific
exoplanet. It creates an instance of the LIFEsim object bus, which serves as the main vehicle
to store all the data related to a LIFEsim simulation. Then it adds all the required modules
to this bus (including the transmission maps, the different photon sources and more) and
imports data from a previously run simulation of n universes as described in Section 1.4.1.
The user can then select the planet number from this imported catalog for which he wishes
to perform the signal extraction as well as additional parameters.
Upon running, the file calls the function single_spectrum_extraction from the main file where
the extraction is carried out. The returned output consists of several plots depicting the transmission maps, the time series of the signal, a heatmap and the distribution of the cost function
J′′, the blackbody curve fits as well as printing out various quantities that are calculated during
the extraction process.
• MultiPlanet_extraction.py: This file is used to perform signal extraction for a large number of exoplanets. Similarly to the previous file, a bus is created serving as the underlying
vehicle and the catalog of a previously carried out MC simulation of n universes is imported.
Instead of choosing one planet, this file uses a matrix called "parameters" that allows the
user to define arbitrary ranges for different parameters with which to filter all of the planets
in the input catalog. For example, one can choose to consider all detected planets with a
radius between 0.75R⊕ −1.5R⊕ and a received stellar insolation between 1SE −3SE. The code
calls the function main_parameter_extraction from the main file, which performs the signal
extraction for each of the planets in the input catalog that fulfill the criteria according to the
parameters matrix. The output is saved in a csv-file for further processing.
• Extraction_DataAnalysis.py: This file is used for post-processing data from a run of
MultiPlanet_extraction.py. It takes the aforementioned csv-file as an input and calculates
aggregated quantities over all considered planets as well as produces plots shown in Chapter
3

Additionally, the LIFEsim core file must be updated with the following ExctractionModule: (see also the thesis)

class ExtractionModule(Module):
'''
Module for extracting planetary parameters from an existing catalog that has already
run through a simulation
'''
def __init__(self,
name: str):
super().__init__(name=name)
self.add_socket(s_name='transmission',
s_type=TransmissionModule,
s_number=1)
self.add_socket(s_name='instrument',
s_type=InstrumentModule,
s_number=1)

@abc.abstractmethod
def single_spectrum_extraction(self):
"""
The single_spectrum_extraction function should return the extracted parameters
like spectra, snr, sigma, position
parameters r (radial) and phi (azimuthal), planet radius, planet temperature (and
potentially additional ones)
"""
pass
