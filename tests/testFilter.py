import unittest

from ssm_ml import Filter


class TestFilter(unittest.TestCase):
    '''
    Test class for the Filter class 
    
    '''
    
    def setUp(self):
        '''
        Create the datasets for testing.
        '''
        
        self.defaultData = [
                {
                    "components" : [
                            {"name" : "foo"},
                            {"name" : "bar"}
                        ],
                    "properties" : {
                            "phase" : "solid",
                            "name" : "hydrogen"
                    },
                    
                    "spectra" : [
                        {
                            "x-axis" : {
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            "y-axis" : {
                                "axis" : "y-axis",
                                "data" : [0, 1, 2, 4, 8, 16, 8, 4, 2, 1]
                                }
                    }
                    ]
                },
                {    
                   "components" : [
                            {"name" : "foo"}
                        ],
                    "properties" : {
                            "phase" : "solid",
                            "name" : "helium"
                    },
                    
                    "spectra" : [
                        {
                            "x-axis" : {
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            "y-axis" : {
                                "axis" : "y-axis",
                                "data" : [0, 1, 2, 4, 15, 7, 8, 4, 2, 1]
                                }
                    }
                        ]
                },
                {
                        "components" : [
                            {"name" : "bar"}
                        ],
                    "properties" : {
                            "phase" : "liquid",
                            "name" : "lithium"
                    },
                    
                    "spectra" : [
                            {
                                "x-axis" : {
                                    "axis" : "x-axis",
                                    "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                                "y-axis" : {
                                    "axis" : "y-axis",
                                    "data" : [0, 1, 2, 4, 8, 10, 12, 14, 16, 18]
                                }
                            }
                            ]
                }
            ]
        
        self.twoRangeData = [
                {
                    "properties" : {
                            "phase" : "solid",
                            "name" : "hydrogen"
                    },
                    
                    "spectra" : [
                        {
                            "x-axis" : {
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            "y-axis" : {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 1, 2, 1, 0]
                                }
                    }
                    ]
                },
                {    
                    "properties" : {
                            "phase" : "solid",
                            "name" : "helium"
                    },
                    
                    "spectra" : [
                        {
                            "x-axis" : {
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            "y-axis" : {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 3, 2, 1, 0 ]
                                }
                    }
                        ]
                },
                {
                    
                    "properties" : {
                            "phase" : "liquid",
                            "name" : "lithium"
                    },
                    
                    "spectra" : [
                        {
                            "x-axis" : {
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            "y-axis" : {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 4, 1, 2, 5]
                                }
                    }
                        ]
                }
            ]
    
    def test_features(self): 
        '''
        Test getting features.
        '''
        
        filter = Filter.Filter()
        filter.featurePaths = [ ["spectra", "SSM:XY:axis:PEAK", "data"],
                               ["spectra", "SSM:XY:axis:PEAK-LOC", "data"],
                               ["spectra", "SSM:XY:axis:PEAK-LOC-RANGE-15-19", "data"],
                               ["spectra", "SSM:XY:axis:PEAK-LOC-RANGE-14.5-29", "data"],
                               ["spectra", "SSM:XY:axis:PEAK-RATIO-RANGE-15-19", "data"]
            ]
        
        features = filter.getFeatures(self.defaultData)
        
        for t in self.twoRangeData:
            print(t['spectra'][0])
        
        # Test that peak finding works
        self.assertEquals(16, features[0][0])
        self.assertEquals(15, features[1][0])
        self.assertEquals(18, features[2][0])
        # self.assertListEqual([16, 15, 18], features[0])
        
        # Test that finding the peak location works
        self.assertEquals(15, features[0][1])
        self.assertEquals(14, features[1][1])
        self.assertEquals(19, features[2][1])
        # self.assertListEqual([15, 14, 19, features[1]])
        
        features = filter.getFeatures(self.twoRangeData)
        
        # Test that peak finding in a range works
        self.assertEquals(17, features[0][2])
        self.assertEquals(16, features[1][2])
        self.assertEquals(19, features[2][2])
        # self.assertListEqual([17, 16, 19], features[2])
        
        # Test that peak finding in a partially overlapping range works
        self.assertEquals(17, features[0][3])
        self.assertEquals(16, features[1][3])
        self.assertEquals(19, features[2][3])
        # self.assertListEqual([17, 16, 19], features[3])
        
        # Test that finding the ratio of the peak over average values works
        self.assertEquals(2.0 / 0.8, features[0][4])
        self.assertEquals(3.0 / 1.2, features[1][4])
        self.assertEquals(5.0 / 2.4, features[2][4])
        # self.assertListEqual([2.0 / 5.0, 6.0 / 1.2, 5.0 / 2.4], features[4])
        
    def test_label(self):
        '''
        Test getting label
        '''  
        
        filter = Filter.Filter()
        
        # Test finding a property in the data
        filter.labelPath = ["properties", "phase"]
        labels = filter.getLabels(self.defaultData)
        self.assertListEqual(["solid", "solid", "liquid"], labels)
        
        # Test finding presense/absense of an item in a list
        filter.labelPath = ["components", "SSM:PRESENT:name:foo"]
        labels = filter.getLabels(self.defaultData)
        self.assertListEqual(["foo_present", "foo_present", "foo_absent"], labels)
        
