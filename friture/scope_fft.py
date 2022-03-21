
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2009 Timoth?Lecomte

# This file is part of Friture.
#
# Friture is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# Friture is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Friture.  If not, see <http://www.gnu.org/licenses/>.

import logging
# import scipy.signal as signal
from friture_extensions.exp_smoothing_conv import pyx_exp_smoothed_value_numpy

from PyQt5 import QtWidgets
from PyQt5.QtQuickWidgets import QQuickWidget

from numpy import log10, where, sign, arange, zeros, ones, sin, array,float64,amax,floor

from friture.store import GetStore
from friture.audiobackend import SAMPLING_RATE
from friture.scope_data import Scope_Data
from friture.curve import Curve
from friture.qml_tools import qml_url, raise_if_error
from friture.audioproc import audioproc  # audio processing class
from friture.spectrum_settings import (Spectrum_Settings_Dialog,  # settings dialog
                                       DEFAULT_FFT_SIZE,
                                       DEFAULT_FREQ_SCALE,
                                       DEFAULT_MAXFREQ,
                                       DEFAULT_MINFREQ,
                                       DEFAULT_SPEC_MIN,
                                       DEFAULT_SPEC_MAX,
                                       DEFAULT_WEIGHTING,
                                       DEFAULT_RESPONSE_TIME,
                                       DEFAULT_SHOW_FREQ_LABELS)



SMOOTH_DISPLAY_TIMER_PERIOD_MS = 25
DEFAULT_TIMERANGE = 2 * SMOOTH_DISPLAY_TIMER_PERIOD_MS
DEFAULT_FREQUENCY1 = 1000
DEFAULT_FREQUENCY2 = 1500

DEFAULT_MIN=-180
DEFAULT_MAX=10


class Scope_Widget1(QtWidgets.QWidget):

    def __init__(self, parent, engine):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)

        self.audiobuffer = None

        store = GetStore()
        self._scope_data = Scope_Data(store)
        store._dock_states.append(self._scope_data)
        state_id = len(store._dock_states) - 1

        self._curve = Curve()
        # self._curve.name = "Ch1 at "+str(DEFAULT_FREQUENCY1)
        self._scope_data.add_plot_item(self._curve)

        self._curve_2 = Curve()
        # self._curve.name = "Ch2 at "+str(DEFAULT_FREQUENCY2)

        self._scope_data.vertical_axis.name = "FFT amplitude (mV)"
        self._scope_data.vertical_axis.setTrackerFormatter(lambda x: "%#.3g" % (x))
        self._scope_data.horizontal_axis.name = "FFT points"
        self._scope_data.horizontal_axis.setTrackerFormatter(lambda x: "%#.3g s" % (x))

        self.setObjectName("Scope_Widget")
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(2, 2, 2, 2)

        self.quickWidget = QQuickWidget(engine, self)
        self.quickWidget.statusChanged.connect(self.on_status_changed)
        self.quickWidget.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.quickWidget.setSource(qml_url("Scope.qml"))
        
        raise_if_error(self.quickWidget)

        self.quickWidget.rootObject().setProperty("stateId", state_id)

        self.gridLayout.addWidget(self.quickWidget)

        self.settings_dialog = Scope_Settings_Dialog(self)

        # self.set_timerange(DEFAULT_TIMERANGE)

        # self.time = zeros(10)
        # self.y = zeros(10)
        # self.y2 = zeros(10)



        # initialize the class instance that will do the fft
        self.proc = audioproc()
        # self.maxfreq = DEFAULT_MAXFREQ
        # self.proc.set_maxfreq(self.maxfreq)
        # self.minfreq = DEFAULT_MINFREQ
        self.fft_size = 2 ** DEFAULT_FFT_SIZE * 32 #8192


        timerange =  1000  # Here in this code is actually the fft points of the fft buffer

        self.set_timerange(timerange)
        self.proc.set_fftsize(self.fft_size)
        # self.spec_min = DEFAULT_SPEC_MIN
        # self.spec_max = DEFAULT_SPEC_MAX
        # self.weighting = DEFAULT_WEIGHTING
        # self.dual_channels = False
        # self.response_time = DEFAULT_RESPONSE_TIME
        self.freq = self.proc.get_freq_scale()

        self.buffersize=1000 #how many fft points to save
        self.buff1=zeros(self.buffersize)
        self.buff2=zeros(self.buffersize)
        self.buff0=zeros(self.buffersize)
        self.buff3=zeros(self.buffersize)

        self.set_frequency1(DEFAULT_FREQUENCY1)
        self.set_frequency2(DEFAULT_FREQUENCY2)

        self.RANGE_MIN=DEFAULT_MIN # self.RANGE_MIN will set the minimum value of the y axis in the GUI
        self.RANGE_MAX=DEFAULT_MAX # unit is volt here
        # self._scope_data.vertical_axis.setRange(1000*self.RANGE_MIN, 1000*self.RANGE_MAX)# Make the unit to be mV
        self._scope_data.vertical_axis.setRange( self.RANGE_MIN, self.RANGE_MAX)# This is simply the range of the label in y axis, has no real relation to the y data.
       
        self.old_index = 0
        self.overlap = 3. / 4.
        self.dual_channels = True

    def on_status_changed(self, status):
        if status == QQuickWidget.Error:
            for error in self.quickWidget.errors():
                self.logger.error("QML error: " + error.toString())

    # method
    def set_buffer(self, buffer):
        self.audiobuffer = buffer
        self.old_index = self.audiobuffer.ringbuffer.offset

    def handle_new_data(self, floatdata):
        
        # floatdata = self.audiobuffer.data(self.fft_size) #This way only the data in the ring buffer is used,
        #which is smaller than the self.fft_size

        #####################
        # we need to maintain an index of where we are in the buffer
        index = self.audiobuffer.ringbuffer.offset

        available = index - self.old_index

        if available < 0:
            # ringbuffer must have grown or something...
            available = 0
            self.old_index = index

        # if we have enough data to add a frequency column in the time-frequency plane, compute it
        needed = self.fft_size * (1. - self.overlap)
        realizable = int(floor(available / needed))

        twoChannels = True
        if realizable > 0:
            sp1n = zeros((len(self.freq), realizable), dtype=float64)
            sp2n = zeros((len(self.freq), realizable), dtype=float64)

            for i in range(realizable):
                floatdata = self.audiobuffer.data_indexed(self.old_index, self.fft_size)

                # first channel
                # FFT transform
                sp1n[:, i] = self.proc.analyzelive(floatdata[0, :])

                if self.dual_channels and floatdata.shape[0] > 1:
                    # second channel for comparison
                    sp2n[:, i] = self.proc.analyzelive(floatdata[1, :])

                self.old_index += int(needed)

            # compute the widget 
            # Kingson: I believe the self.dispbuffers below are used to display the fading red plot in the graph
            # sp1 = pyx_exp_smoothed_value_numpy(self.kernel, self.alpha, sp1n, self.dispbuffers1)
            # sp2 = pyx_exp_smoothed_value_numpy(self.kernel, self.alpha, sp2n, self.dispbuffers2)
            # # store result for next computation
            # self.dispbuffers1 = sp1
            # self.dispbuffers2 = sp2


        #############################


        # time = self.timerange * 1e-3
        # width = int(time * SAMPLING_RATE)
        # # basic trigger capability on leading edge
        # floatdata = self.audiobuffer.data(2 * width)

            # twoChannels = False
            if floatdata.shape[0] > 1:
                twoChannels = True

            if twoChannels and len(self._scope_data.plot_items) == 1:
                self._scope_data.add_plot_item(self._curve_2)
            elif not twoChannels and len(self._scope_data.plot_items) == 2:
                self._scope_data.remove_plot_item(self._curve_2)

            # # trigger on the first channel only
            # triggerdata = floatdata[0, :]
            # # trigger on half of the waveform
            # trig_search_start = width // 2
            # trig_search_stop = -width // 2
            # triggerdata = triggerdata[trig_search_start: trig_search_stop]

            # trigger_level = floatdata.max() * 2. / 3.
            # trigger_pos = where((triggerdata[:-1] < trigger_level) * (triggerdata[1:] >= trigger_level))[0]

            # if len(trigger_pos) == 0:
            #     return

            # if len(trigger_pos) > 0:
            #     shift = trigger_pos[0]
            # else:
            #     shift = 0
            # shift += trig_search_start

            # datarange = width
            # floatdata = floatdata[:, shift - datarange // 2: shift + datarange // 2] # the number of elements in floatdata become datarange here. select the portion of data that meet the trigger condition

            # self.y = floatdata[0, :]
            # if twoChannels:
            #     self.y2 = floatdata[1, :]
            # else:
            #     self.y2 = None


            # sp1n = zeros(self.fft_size, dtype=float64)
            # sp2n = zeros(self.fft_size, dtype=float64)
            # sp1n = self.proc.analyzelive(floatdata[0, :])
            # if twoChannels:
            #     # second channel for comparison
            #     sp2n = self.proc.analyzelive(floatdata[1, :])

            # if twoChannels:
            #     dB_spectrogram =  self.log_spectrogram(sp1n)
            #     dB_spectrogram2= self.log_spectrogram(sp2n)
            # else:
            #     dB_spectrogram = self.log_spectrogram(sp1n) 

        ###########################################################################
            self.freq1=self.frequency1 # frequency I am interested in to extract fft amp ####
            self.freq2=self.frequency2                                                   ####
        ###########################################################################
            self.freq_idx1=(abs(self.freq-self.freq1)).argmin()
            self.freq_idx2=(abs(self.freq-self.freq2)).argmin()
            #check self.freq[self.freq_idx1] , see if it is close to 1000

            data=sp1n[self.freq_idx1]
            data=self.log_spectrogram(data)
            

            self.buff0=self.buff1
            self.buff1[-1]=data
            self.buff1[:-1]=self.buff0[1:]
            # for i in range(len(self.buff1)-1):
            #     self.buff1[i]=self.buff0[i+1]

            b=self.buff1
            a=arange(self.buffersize)
            
            # b_min=min(b)
            # b_max=max(b)
            
            #Kingson: trying to plot in autoscale in y axis. 
            # The y axis ticker is set in above in this function: self._scope_data.vertical_axis.setRange
            # self.RANGE_MIN=b_min*1.2 
            # self.RANGE_MAX=b_min*1.2
            # self._scope_data.vertical_axis.setRange(1000*self.RANGE_MIN, 1000*self.RANGE_MAX)# Make the unit to be mV



            scaled_a=a/self.buffersize

            range_min=self.RANGE_MIN #the minimum value that the target signal can reach at target frequency
            range_max=self.RANGE_MAX

            range_middle=(range_min+range_max)/2
            range_length=range_max-range_min

            b=(b-range_middle)/(range_length/2)  #turn b into the range (-1, 1)

            scaled_b=1-(b+1)/2.  #turn scaled_b into the range (1,0)

            # Design the Butterworth filter using 
            # signal.butter and output='sos'
            # fs = self.buffersize
            # sos = signal.butter(3, 100, 'lp', fs=3000, output='sos') 
            # #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

            # # Filter the signal by the filter using signal.sosfilt
            # # Use signal.sosfiltfilt to get output inphase with input
            # filtered = signal.sosfiltfilt(sos, scaled_b)
            # self._curve.setData(scaled_a, filtered)


            self._curve.setData(scaled_a, scaled_b)
#####################################################
            if twoChannels:
                data2=sp2n[self.freq_idx2]
                data2=self.log_spectrogram(data2)

                self.buff2=self.buff3
                self.buff3[-1]=data2
                # for i in range(len(self.buff3)-1):
                #     self.buff3[i]=self.buff2[i+1]
                self.buff3[:-1]=self.buff2[1:]

                b=self.buff3
                a=arange(self.buffersize)

                scaled_a=a/self.buffersize

                b=(b-range_middle)/(range_length/2)  #turn b into the range (-1, 1)
                scaled_b=1-(b+1)/2.

                self._curve_2.setData(scaled_a, scaled_b)

            # filtered = signal.sosfiltfilt(sos, scaled_b)
            # self._curve_2.setData(scaled_a, filtered)


        # dBscope = False
        # if dBscope:
        #     dBmin = -50.
        #     self.y = sign(self.y) * (20 * log10(abs(self.y))).clip(dBmin, 0.) / (-dBmin) + sign(self.y) * 1.
        #     if twoChannels:
        #         self.y2 = sign(self.y2) * (20 * log10(abs(self.y2))).clip(dBmin, 0.) / (-dBmin) + sign(self.y2) * 1.
        #     else:
        #         self.y2 = None

        # self.time = (arange(len(self.y)) - datarange // 2) / float(SAMPLING_RATE)
        """
        datarange=240, datarange=width, width is number of samples.
                time = self.timerange * 1e-3  #self.timerange is the time set by user in ms.
                width = int(time * SAMPLING_RATE)

        len(self.y)=240
        SAMPLING_RATE=48000

        """


################################################Kingson*************
        #The _curve function will only plot the data with x in the range of 0 to 1, and y in the range of -1 to 1, following code showed such an arrangement:
        # a=array([0, 0.5, 1])
        # b=array([-1,0.4, 1])
        # b=1-(b+1)/2.
        # self._curve.setData(a, b)
###############################################Kingson **************


    # def log_spectrogram(self, sp):
    #     # Note: implementing the log10 of the array in Cython did not bring
    #     # any speedup.
    #     # Idea: Instead of computing the log of the data, I could pre-compute
    #     # a list of values associated with the colormap, and then do a search...
    #     # epsilon = 1e-30
    #     # return 10. * log10(sp + epsilon)
    #     return sp


    # method
    def canvasUpdate(self):
        return

    def pause(self):
        return

    def restart(self):
        return

    # slot
    def set_timerange(self, timerange):
        self.timerange = timerange
        self._scope_data.horizontal_axis.setRange(0, self.timerange)
        # self._scope_data.horizontal_axis.setRange(-self.timerange/2., self.timerange/2.)
        self._scope_data.vertical_axis.setRange(0., 1.)


    def set_yrange(self, ymin, ymax):
        # change the y axis tickers in the GUI
        self.RANGE_MIN=ymin # self.RANGE_MIN will set the minimum value of the y axis in the GUI
        self.RANGE_MAX=ymax # unit is volt here, 1.2 here is to make sure the data peaks are not cropped
        self._scope_data.vertical_axis.setRange(self.RANGE_MIN, self.RANGE_MAX)# Make the unit in the plot to be mV


        # Make sure the output data points are scaled properly to match the y axis values.


    # slot
    def set_frequency1(self, frequency):
        self.frequency1 = frequency
        self._curve.name = "Ch1 at "+str(self.frequency1) +" Hz"
        # self._scope_data.horizontal_axis.setRange(0, self.timerange)

    def set_frequency2(self, frequency):
        self.frequency2 = frequency
        self._curve_2.name = "Ch2 at "+str(self.frequency2) +" Hz"
        # self._scope_data.horizontal_axis.setRange(0, self.timerange)

    def setmin(self, value):
        self.RANGE_MIN = value # dB
        self.set_yrange(self.RANGE_MIN, self.RANGE_MAX)

    def setmax(self, value):
        self.RANGE_MAX = value  # dB
        self.set_yrange(self.RANGE_MIN, self.RANGE_MAX)


    def log_spectrogram(self, sp):
        # Note: implementing the log10 of the array in Cython did not bring
        # any speedup.
        # Idea: Instead of computing the log of the data, I could pre-compute
        # a list of values associated with the colormap, and then do a search...
        epsilon = 1e-30
        return 10. * log10(sp + epsilon)


    # slot
    def settings_called(self, checked):
        self.settings_dialog.show()

    # method
    def saveState(self, settings):
        self.settings_dialog.saveState(settings)

    # method
    def restoreState(self, settings):
        self.settings_dialog.restoreState(settings)


class Scope_Settings_Dialog(QtWidgets.QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("FFT_Points Scope settings")

        self.formLayout = QtWidgets.QFormLayout(self)

        self.doubleSpinBox_frequency1 = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_frequency1.setDecimals(0)
        self.doubleSpinBox_frequency1.setMinimum(20)
        self.doubleSpinBox_frequency1.setMaximum(20000)
        self.doubleSpinBox_frequency1.setProperty("value", DEFAULT_FREQUENCY1)
        self.doubleSpinBox_frequency1.setObjectName("doubleSpinBox_frequency1")
        self.doubleSpinBox_frequency1.setSuffix(" Hz")

        self.doubleSpinBox_frequency2 = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_frequency2.setDecimals(0)
        self.doubleSpinBox_frequency2.setMinimum(20)
        self.doubleSpinBox_frequency2.setMaximum(20000)
        self.doubleSpinBox_frequency2.setProperty("value", DEFAULT_FREQUENCY2)
        self.doubleSpinBox_frequency2.setObjectName("doubleSpinBox_frequency2")
        self.doubleSpinBox_frequency2.setSuffix(" Hz")

        self.doubleSpinBox_min = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_min.setDecimals(1)
        self.doubleSpinBox_min.setMinimum(-220)
        self.doubleSpinBox_min.setMaximum(10)
        self.doubleSpinBox_min.setProperty("value", DEFAULT_MIN)
        self.doubleSpinBox_min.setObjectName("doubleSpinBox_min")
        self.doubleSpinBox_min.setSuffix(" dB")

        self.doubleSpinBox_max = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_max.setDecimals(1)
        self.doubleSpinBox_max.setMinimum(-220)
        self.doubleSpinBox_max.setMaximum(10)
        self.doubleSpinBox_max.setProperty("value", DEFAULT_MAX)
        self.doubleSpinBox_max.setObjectName("doubleSpinBox_max")
        self.doubleSpinBox_max.setSuffix(" dB")

        # self.formLayout.addRow("Time range:", self.doubleSpinBox_timerange)
        self.formLayout.addRow("Target frequency at Ch1:", self.doubleSpinBox_frequency1)
        self.formLayout.addRow("Target frequency at Ch2:", self.doubleSpinBox_frequency2)
        self.formLayout.addRow("Min level: ", self.doubleSpinBox_min)
        self.formLayout.addRow("Max level: ", self.doubleSpinBox_max)

        self.setLayout(self.formLayout)

        # self.doubleSpinBox_timerange.valueChanged.connect(self.parent().set_timerange)
        self.doubleSpinBox_frequency1.valueChanged.connect(self.parent().set_frequency1)
        self.doubleSpinBox_frequency2.valueChanged.connect(self.parent().set_frequency2)

        self.doubleSpinBox_min.valueChanged.connect(self.parent().setmin)
        self.doubleSpinBox_max.valueChanged.connect(self.parent().setmax)


    # method
    def saveState(self, settings):
        # settings.setValue("timeRange", self.doubleSpinBox_timerange.value())
        settings.setValue("frequency1", self.doubleSpinBox_frequency1.value())
        settings.setValue("frequency2", self.doubleSpinBox_frequency2.value())

        settings.setValue("RANGE_MIN", self.doubleSpinBox_min.value())
        settings.setValue("RANGE_MAX", self.doubleSpinBox_max.value())

    # method
    def restoreState(self, settings):
        # timeRange = settings.value("timeRange", DEFAULT_TIMERANGE, type=float)
        # self.doubleSpinBox_timerange.setValue(timeRange)

        frequency1 = settings.value("frequency1", DEFAULT_FREQUENCY1, type=float)
        self.doubleSpinBox_frequency1.setValue(frequency1)

        frequency2 = settings.value("frequency2", DEFAULT_FREQUENCY2, type=float)
        self.doubleSpinBox_frequency2.setValue(frequency2)

        RANGE_MIN = settings.value("RANGE_MIN", DEFAULT_MIN, type=float)
        self.doubleSpinBox_min.setValue(RANGE_MIN)

        RANGE_MAX = settings.value("RANGE_MAX", DEFAULT_MAX, type=float)
        self.doubleSpinBox_max.setValue(RANGE_MAX)