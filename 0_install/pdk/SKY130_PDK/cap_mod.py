import math
from align.primitive.default.canvas import DefaultCanvas
from align.cell_fabric.generators import *
from align.cell_fabric.grid import *

import logging
logger = logging.getLogger(__name__)

class CapGenerator(DefaultCanvas):

    def __init__(self, pdk):
        super().__init__(pdk)
        
        self.boundary_offset = -100
        
        self.m5n = self.addGen(Wire( 'm5n_plate', 'M5', 'v',
                    clg=UncoloredCenterLineGrid( pitch=self.pdk['M5']['Pitch'], width=self.pdk['Cap']['m5Width']+400, offset=self.boundary_offset),
                    spg=EnclosureGrid(pitch=self.pdk['M4']['Pitch'], stoppoint=0, offset=-self.pdk['M4']['Width']//4 - 2000, check=False)))
 
        self.m3n = self.addGen( Wire( 'm3n', 'CapMIMLayer', 'v',
                                    clg=UncoloredCenterLineGrid( pitch=self.pdk['M3']['Pitch'], width=self.pdk['M3']['Width']),
                                    spg=EnclosureGrid(pitch=self.pdk['M2']['Pitch'], stoppoint=self.pdk['V2']['VencA_H'] + self.pdk['M2']['Width']//2, check=False)))
        
        self.m5_offset = self.pdk['CapMIMLayer']['Enclosure'] + self.pdk['CapMIMContact']['Enclosure'] + self.pdk['CapMIMContact']['WidthX']//2
        self.m4n = self.addGen(Wire( 'm4n', 'M4', 'v',
                                     clg=UncoloredCenterLineGrid( pitch=2*self.pdk['Cap']['m5Width'], width=self.pdk['Cap']['m5Width']+400, offset=self.m5_offset-1700+self.boundary_offset),
                                     spg=EnclosureGrid(pitch=self.pdk['M4']['Pitch']//2, stoppoint=self.pdk['CapMIMContact']['Enclosure'], offset=0, check=False)))

        self.Cboundary = self.addGen( Region( 'Cboundary', 'Cboundary', h_grid=self.m2.clg, v_grid=self.m1.clg))


        h_clg_mim = UncoloredCenterLineGrid( pitch=2, width=2)
        v_clg_mim = UncoloredCenterLineGrid( pitch=2, width=2, offset=self.m5_offset-1250)

        self.CapMIMC = self.addGen( Region( 'CapMIMC', 'CapMIMContact', h_grid=h_clg_mim, v_grid=v_clg_mim))
        #self.CapV3 = self.addGen( Region( 'CapV3', 'CapV3', h_grid=h_clg_mim, v_grid=v_clg_mim))
        #self.CapV2 = self.addGen( Region( 'CapV2', 'CapV2', h_grid=h_clg_mim, v_grid=v_clg_mim))
        
        #self.v4_x = self.addGen( Via( 'v4_x', 'V4',
        #                                h_clg=self.m4.clg, v_clg=self.m5.clg,
        #                                WidthX=self.v4.WidthX, WidthY=self.v4.WidthY,
        #                                h_ext=self.v4.h_ext, v_ext=self.v4.v_ext))
        #
        #self.v3_x = self.addGen( Via( 'v3_x', 'V3',
        #                                h_clg=self.m3.clg, v_clg=self.m4.clg,
        #                                WidthX=self.v3.WidthX, WidthY=self.v3.WidthY,
        #                                h_ext=self.v3.h_ext, v_ext=self.v3.v_ext))
        #
        #self.v2_x = self.addGen( Via( 'v2_x', 'V2',
        #                                h_clg=self.m2.clg, v_clg=self.m3.clg,
        #                                WidthX=self.v2.WidthX, WidthY=self.v2.WidthY,
        #                                h_ext=self.v2.h_ext, v_ext=self.v2.v_ext))

    def addCap( self, length, width):
        x_length = int(length)
        y_length = int(width)

        m1_p = self.pdk['M1']['Pitch']
        m2_p = self.pdk['M2']['Pitch']

        m4n_xwidth = x_length + 2*self.pdk['CapMIMLayer']['Enclosure']
        
        mim_layer = Wire( 'mim_layer', 'CapMIMLayer', 'v',
                    clg=UncoloredCenterLineGrid( pitch=2*m4n_xwidth, width=m4n_xwidth, offset=m4n_xwidth//2+self.boundary_offset),
                    spg=EnclosureGrid(pitch=y_length, stoppoint=self.pdk['CapMIMLayer']['Enclosure'], check=False))
        
        m5n_plate = Wire( 'm5n', 'M5', 'v',
                        clg=UncoloredCenterLineGrid( pitch=2*x_length, width=x_length, offset=x_length//2+self.pdk['CapMIMLayer']['Enclosure']+self.boundary_offset),
                        spg=EnclosureGrid(pitch=y_length, stoppoint=0, check=False))
        
        #m3n = Wire( 'm3n', 'M4', 'v',
        #            clg=UncoloredCenterLineGrid( pitch=2*m4n_xwidth, width=1.08*m4n_xwidth, offset=m4n_xwidth//2 - 500+self.boundary_offset),
        #            spg=EnclosureGrid(pitch=1.04*y_length, stoppoint=self.pdk['CapMIMLayer']['Enclosure']*4, check=False))
        
        m4n = Wire( 'm4n', 'M4', 'v',
                    clg=UncoloredCenterLineGrid( pitch=2*m4n_xwidth, width=m4n_xwidth*1.1, offset=m4n_xwidth//2+self.boundary_offset),
                    spg=EnclosureGrid(pitch=y_length*1.1, stoppoint=self.pdk['CapMIMLayer']['Enclosure'], check=False, offset=-2000))


        x_number = math.ceil(m4n_xwidth/m1_p)
        y_number_m4 = math.ceil((y_length+self.pdk['CapMIMLayer']['Enclosure']+0.5*self.pdk['M4']['Width'])/self.pdk['M4']['Pitch'])
        y_number = math.ceil((y_number_m4*self.pdk['M4']['Pitch'])/m2_p)

        logger.debug( f"Number of wires {x_number} {y_number}")
        
        # Metal 4
        self.addWire( self.m5n, 'PLUS', 1, (y_number_m4-1-1, -1), (y_number_m4+2, 7), netType='pin')
        self.addWire( m5n_plate, 'PLUS', 0, (0, -1), (1, 1))
        
        # Capm
        self.addWire( mim_layer, 'Capm', 0, (0, -1), (1, 1))
        
        # Metal 3
        self.addWire( self.m4n, 'MINUS', 0, (-3, -3), (1, 1), netType='pin')
        self.addWire( m4n, 'MINUS', 0, (0, -1), (1, 1))
        
        gridx0= (self.m5_offset - self.pdk['CapMIMContact']['WidthX']//2)//2
        gridx1= gridx0 + self.pdk['CapMIMContact']['WidthX']//2
        #offset_y = -480
        #offset_x = -100
        #self.addRegion( self.CapV3, None, gridx0+offset_x, 150, gridx1+offset_x, 250)
        #self.addRegion( self.CapV2, None, gridx0+offset_x, -100+offset_y, gridx1+offset_x, -200+offset_y)
        ###self.addVia( self.v3_x, None, -1, +1)
        ###self.addVia( self.v2_x, None, -3, -3)
        offset_x = 850 #16700
        offset_y = 15500 #19300
        self.addRegion( self.CapMIMC, None, gridx0+offset_x, 150+offset_y, gridx1+offset_x, 250+offset_y)
        ###self.addVia( self.v4_x, None, 22, 31)
        #gridx2 = math.floor(m4n_xwidth/self.pdk['M3']['Pitch'])
        ###self.addWire( self.m4, 'PLUS', y_number_m4+3, (0, -21), (gridx2, 7), netType = 'pin')
        ###self.addWire( self.m2, 'MINUS', -3, (0, -17), (gridx2, 1), netType = 'pin')
 
        self.addRegion( self.boundary, 'Boundary', -6, -6,
                        x_number+4,
                        y_number+15)

        #self.addRegion( self.Cboundary, 'Cboundary',
        #                -4, -4,
        #                x_number+5,
        #                y_number+13)

        logger.debug( f"Computed Boundary: {self.terminals[-1]} {self.terminals[-1]['rect'][2]} {self.terminals[-1]['rect'][2]%80}")
