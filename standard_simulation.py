import lifesim as ls
import numpy as np
from auxiliary import path
from Extraction import ML_Extraction

#create bus
bus=ls.Bus()

#set basic scenario and make potential changes with .set_manual
bus.data.options.set_scenario('baseline')
#bus.data.options.set_manual(diameter=4.)

#import ppop data to bus
#bus.data.catalog_from_ppop(input_path=path+'04_input_files/ppop_catalog_10.txt')
bus.data.catalog_from_ppop(input_path=path+'04_input_files/ppop_catalog_500.fits')

#remove potential stars from the bus; removing A stars and (M stars > 10parsec) is assumed in the standard scenarios 1&2
bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)
bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.)


#add the instrument, transmission and noise modules and connect them
instrument = ls.Instrument(name='inst')
bus.add_module(instrument)

transm = ls.TransmissionMap(name='transm')
bus.add_module(transm)

exozodi = ls.PhotonNoiseExozodi(name='exo')
bus.add_module(exozodi)

localzodi = ls.PhotonNoiseLocalzodi(name='local')
bus.add_module(localzodi)

star_leak = ls.PhotonNoiseStar(name='star')
bus.add_module(star_leak)

extr = ML_Extraction(name='extr')
bus.add_module(extr)

bus.connect(('inst', 'transm'))
bus.connect(('inst', 'exo'))
bus.connect(('inst', 'local'))
bus.connect(('inst', 'star'))
bus.connect(('star', 'transm'))

bus.connect(('extr', 'inst'))
bus.connect(('extr', 'transm'))


#add and connect the optimizer and ahgs modules
opt = ls.Optimizer(name='opt')
bus.add_module(opt)
ahgs = ls.AhgsModule(name='ahgs')
bus.add_module(ahgs)

bus.connect(('transm', 'opt'))
bus.connect(('inst', 'opt'))
bus.connect(('opt', 'ahgs'))


#run simulation (this is where the stuff happens) ---------------------------------------------
#get SNR for 1 hour for every planet created by PPOP
instrument.get_snr(save_mode=True)

#optimize the search phase; set ...['habitable'] to True/False to optimize for # of planets in habitable zone
bus.data.options.optimization['habitable'] = True
opt.ahgs()

#saving the results
bus.data.export_catalog(output_path=path+'05_output_files/changeme.hdf5')
bus.data.catalog.to_csv(path+'05_output_files/changeme.csv')