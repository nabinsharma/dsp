import numpy
import pylab
import scipy.io.wavfile
import scipy.signal

import utils

def short_time_energy_and_zero_cross_rate():
  """Example: Computation of ST-ZCR and STE of a speech signal."""

  fs, x = scipy.io.wavfile.read("data/so.wav")
  x = numpy.array(x, dtype=float)
  t = numpy.arange(len(x)) * (1.0 / fs)
  
  # Find the short time zero crossing rate.
  zc = utils.stzcr(x, scipy.signal.get_window("boxcar", 201))
  
  # Find the short time energy.
  e = utils.ste(x, scipy.signal.get_window("hamming", 201))
  
  pylab.figure()
  pylab.subplot(311)
  pylab.plot(t, x)
  pylab.title('Speech signal (so.wav)')
  pylab.subplot(312)
  pylab.plot(t, zc, 'r', linewidth=2)
  pylab.title('Short-time Zero Crossing Rate')
  pylab.subplot(313)
  pylab.plot(t, e, 'm', linewidth=2)
  pylab.xlabel('t (s)')
  pylab.title('Short-time Energy')
  
  pylab.figure()
  pylab.plot(t, x / x.max(), label="so.wav")
  pylab.hold(True)
  pylab.plot(t, zc / zc.max(), 'r', label="zero crossing rate")
  pylab.plot(t, e / e.max(), 'm', label="energy")
  pylab.legend()


def run_examples():
  short_time_energy_and_zero_cross_rate()
  pylab.show()


if __name__ == "__main__":
  run_examples()
