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

from numpy import zeros

class bridge():
   
    def __init__(self):
        super().__init__()

        self.buff1=zeros(100) # save the fft amp to be ploted
        self.buff2=zeros(100)


    def read(self):
        return self.buff1



    def write(self, data):
        self.buff2=self.buff1
        self.buff1[-1]=data
        for i in range(len(self.buff1)-1):
            self.buff1[i]=self.buff2[i+1]
