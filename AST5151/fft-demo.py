#! /usr/bin/env python

# AST 5151 - Principles of Planetary Physics
# Math module
# Joseph Harrington
# Demo: Fourier transforms

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

plt.ion()

n  = 1024
dt = 1
t  = np.arange(n)

# simple sinusoidal signal
st = np.sin(2. * pi * t/n)
plt.plot(t)

plt.clf()
plt.plot(t, st)
plt.xlabel('Sample Number')
plt.ylabel('Time')
plt.title('Times of Samples')

# n=8
# NOTE:  I'm going to do the example using the full, unoptimized
# fft(), so you can see how the frequencies fold and that the results
# are symmetric.  However, you can use rfft() and avoid all this
# baloney.
ft = np.fft.fft(st)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)
# look at frequencies
plt.clf()
plt.plot(ff, '.')
plt.xlabel('Channel Number')
plt.ylabel('Frequency')
plt.title('Frequency of Powers')
print(ff[0])		# DC term: no cycles per timestep
print(ff[1])		# low frequency: almost no cycles per timestep
print(ff[n-1])		# same, but negative
print(ff[n-2])
print(ff[int(n/2)])		# Nyquist: half a cycle per timestep
print(ff[int(n/2)-1])
print(ff[int(n/2)+1])

rft   = np.real(ft)	# real part
ift   = np.imag(ft)	# imaginary part
ampft = np.abs(ft) / n	# amplitude
phft  = np.arctan2(ift, rft)	# phase
psft  = ampft**2		# power spectrum

# real part
plt.clf()
plt.plot(ff, rft)		# note: LOW numbers
plt.xlabel('Frequency')
plt.ylabel('Real Amplitude')
plt.title('Real FFT')

# imaginary part
plt.clf()
plt.plot(ff, ift)		# there's some power
plt.xlabel('Frequency')
plt.ylabel('Imag. Amplitude')
plt.title('Imag. FFT')

# amplitude spectrum
plt.clf()
plt.plot(ff, ampft, '.')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('FFT')
print(ampft[0:10])	# power is split, half is in negative frequencies
# stretch to see middle

# power spectrum
plt.clf()
plt.plot(ff, psft, '.')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('FFT')

# phases
plt.clf()
plt.plot(ff, phft)
plt.xlabel('Frequency')
plt.ylabel('Phase, [0,2pi]')
plt.title('FFT')
print(phft[0:10])

# phases going [-1,1] instead of [0,2pi]
plt.clf()
plt.plot(ff, phft/(pi), '.')
plt.xlabel('Frequency')
plt.ylabel('Phase, [-1,1]')
plt.title('FFT')
# stretch to see middle

# Where is there actually power?  Phases of others don't matter.
plt.clf()
plt.plot(ff, ampft, '.')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('FFT')

# ok, let's play!

# first, we're dealing with real numbers, so we'll just use the
# positive frequencies

noise  = np.random.normal(size=n)
signal = 0.5 * np.sin(7. * 2. * pi * t/n)
sn     = signal + noise

plt.clf()
plt.plot(t, signal)

plt.clf()
plt.plot(t, noise)

plt.clf()
plt.plot(t, sn)

ft = np.fft.fft(sn)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)
pft    = ft[0:int(n/2)]
pff    = ff[0:int(n/2)]
rpft   = np.real(pft)
ipft   = np.imag(pft)
amppft = np.abs(pft) / n
phpft  = np.arctan2(ipft, rpft)
pspft  = amppft**2
norm   = pspft[0]	 # mean of dataset squared
psd    = pspft/norm # power spectral density

plt.clf()
plt.plot(pff, pspft) # FINDME,        xrange=[-0.025,.525], /xstyle)
plt.clf()
plt.plot(pff, pspft) # FINDME,        xrange=[-0.025,.200], /xstyle)
plt.xlim([-0.025,.2])

print(pspft[0:10])	# channel 7 is high, and that's the sine frequency

# now make the signal small: 1/5 sigma!

signal = 0.2 * np.sin(7. * 2. * pi * t/n)

sn    = signal + noise
plt.clf()
plt.plot(t, signal)

plt.clf()
plt.plot(t, noise)

plt.clf()
plt.plot(t, sn)

ft = np.fft.fft(sn)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)
pft    = ft[0:int(n/2)]
pff    = ff[0:int(n/2)]

# everything we care about
rpft   = np.real(pft)
ipft   = np.imag(pft)
amppft = np.abs(pft) / n
phpft  = np.arctan2(ipft, rpft)
pspft  = amppft**2
norm   = pspft[0]
psd    = pspft/norm

plt.clf()
plt.plot(pff, pspft)

plt.clf()
plt.plot(pff, pspft)
plt.xlim([-0.025,0.1])

print(pspft[0:10])	# channel 7 is high, and that's the sine frequency

# now add a second weak signal

signal =  0.2 * np.sin( 7. * 2. * pi * t/n) \
        + 0.2 * np.sin(12. * 2. * pi * t/n)

sn    = signal + noise
plt.clf()
plt.plot(t, signal)

plt.clf()
plt.plot(t, noise)

plt.clf()
plt.plot(t, sn)
plt.plot(t, signal)


ft = np.fft.fft(sn)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)
pft    = ft[0:int(n/2)]
pff    = ff[0:int(n/2)]
rpft   = np.real(pft)
ipft   = np.imag(pft)
amppft = np.abs(pft) / n
phpft  = np.arctan2(ipft, rpft)
pspft  = amppft**2
norm   = pspft[0]
psd    = pspft/norm

plt.clf()
plt.plot(pff, pspft)

plt.clf()
plt.plot(pff, pspft)
plt.xlim([-0.025,0.1])
print(pspft[0:15])	# channels 7 and 12 high

# how about a non-integer sinusoid?

signal = np.sin( 33.65 * 2. * pi * t/n)

plt.clf()
plt.plot(t, signal)


ft = np.fft.fft(signal)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)
pft    = ft[0:int(n/2)]
pff    = ff[0:int(n/2)]

# everything we care about
rpft   = np.real(pft)
ipft   = np.imag(pft)
amppft = np.abs(pft) / n
phpft  = np.arctan2(ipft, rpft)
pspft  = amppft**2
norm   = amppft[0]
psd    = pspft/norm

plt.clf()
plt.plot(pff, pspft)

plt.clf()
plt.plot(pff, pspft)
plt.xlim([-0.025,0.1])

plt.clf()
plt.plot(pff, pspft, '.')
plt.xlim([-0.01,0.05])

# periodic signals

dt = 1
persig = np.array([10., 23, 32, 19, 87, 90, 93, 91, 83, 65, 32, 14, 12, 13, 11])
plt.clf()
plt.plot(persig)

# repeat that 30 times (using a Python broadcasting trick)
nrep = 30
signal = np.zeros((nrep, persig.size))
signal += persig
signal = signal.flatten()

# truncate that
n      = persig.size * nrep - 6
signal = signal[0 : n]
t      = np.arange(n)

plt.clf()
plt.plot(t, signal)


ft = np.fft.fft(signal)
ff = np.concatenate( (np.arange(n/2+1.), -(np.arange(n/2-1.) + 1)[::-1]) ) / (n*dt)

# everything we care about
rft   = np.real(ft)	# real part
ift   = np.imag(ft)	# imaginary part
ampft = np.abs(ft) / n	# amplitude
phft  = np.arctan2(ift, rft)	# phase
psft  = ampft**2		# power spectrum

plt.clf()
plt.plot(ff, ampft)     # Harmonics!

print(signal.mean())
print(ampft[0])         # Yup, channel 0 is the data mean.

newt = t.copy()

newsig = 0
for k in np.arange(n):
  newsig += ampft[k] * np.sin(2. * pi * k * newt / n + phft[k] + pi/2.)

plt.clf()
plt.plot(   t, signal)
plt.plot(newt, newsig)
plt.xlim([0,30])

plt.clf()
plt.plot(newt, signal-newsig)

print(np.min(signal-newsig))
print(np.max(signal-newsig))

# approximate with just first 125 terms

newt = t.copy()

newsig = ampft[0]
for k in np.arange(125) + 1:
  newsig =  newsig \
          + 2*ampft[k] * np.sin(2. * pi * k * newt / n + phft[k] + pi/2.)

plt.clf()
plt.plot(   t, signal)
plt.plot(newt, newsig)

plt.clf()
plt.plot(   t, signal)
plt.plot(newt, newsig)
plt.xlim([0,30])

plt.clf()
plt.plot(newt, signal-newsig)

print(np.min(signal-newsig))
print(np.max(signal-newsig))

# create synthetic signal with 5.3 times more points, for interpolation

newt = np.arange(n * 5.3 ) / 5.3

newsig = ampft[0]
for k in np.arange(125)+1:
  newsig =  newsig \
          + 2*ampft[k] * np.sin(2. * pi * k * newt / n + phft[k] + pi/2.)

plt.clf()
plt.plot(t, signal)
plt.plot(newt, newsig)
plt.xlim([0,30])

# periodogram
# Fold with a given period, adjust the period to minimize spread in points.
# An FFT gives a good first guess as to the folding period.

pert = t % 10.3
plt.clf()
plt.plot(pert, signal, '.')

pert = t % 14.9
plt.clf()
plt.plot(pert, signal, '.')

pert = t % 15.1
plt.clf()
plt.plot(pert, signal, '.')

pert = t % 14.97
plt.clf()
plt.plot(pert, signal, '.')

pert = t % 14.99
plt.clf()
plt.plot(pert, signal, '.')

pert = t % 15
plt.clf()
plt.plot(pert, signal, '.')
