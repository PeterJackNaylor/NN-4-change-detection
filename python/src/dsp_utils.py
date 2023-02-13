import numpy as np


def powerspec(data, nfft=None, verbose=False):
    if verbose: print ("shape of data = ", data.shape)

    if nfft is None:
        nfft = np.sqrt(data.size)

    eps = 1e-50 # to void log(0)
    amplsU = np.fft.fft2(data, s=[nfft, nfft])/data.size

    Ek  = np.abs(amplsU)**2
    Ek = np.fft.fftshift(Ek)

    sign_sizex = np.shape(Ek)[0]
    sign_sizey = np.shape(Ek)[1]

    box_sidex = sign_sizex
    box_sidey = sign_sizey

    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)

    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)

    if verbose:
        print ("box sidex     =",box_sidex) 
        print ("box sidey     =",box_sidey) 
        print ("sphere radius =",box_radius )
        print ("centerbox     =",centerx)
        print ("centerboy     =",centery)

    Ek_avsphr = np.zeros(box_radius,)+eps ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            Ek_avsphr[wn] = Ek_avsphr [wn] + Ek [i,j]

    return Ek_avsphr