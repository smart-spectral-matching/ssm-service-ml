import pytest

from ssm_ml import Filter


@pytest.fixture   
def default_data():       
    data = [
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
                    }
                },
                {
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
                    }
                },
                {
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
                    }
                },
                {
                    "y-axis" : {
                        "axis" : "y-axis",
                        "data" : [0, 1, 2, 4, 8, 10, 12, 14, 16, 18]
                    }
                }
            ]
        }
    ]
    return data


@pytest.fixture
def two_range_data():
    data = [
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
                    }
                },
                {
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
                    }
                },
                {
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
                    }
                },
                {
                    "y-axis" : {
                        "axis" : "y-axis",
                        "data" : [21, 22, 24, 22, 21, 0, 4, 1, 2, 5]
                    }
                }
            ]
        }
    ]
    return data

def test_features(default_data, two_range_data): 
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
    
    features = filter.getFeatures(default_data)
    
    for t in two_range_data:
        print(t['spectra'][0])
    
    # Test that peak finding works
    assert 16 == features[0][0]
    assert 15 == features[1][0]
    assert 18 == features[2][0]
    
    # Test that finding the peak location works
    assert 15 == features[0][1]
    assert 14 == features[1][1]
    assert 19 == features[2][1]
    
    features = filter.getFeatures(two_range_data)
    
    # Test that peak finding in a range works
    assert 17 == features[0][2]
    assert 16 == features[1][2]
    assert 19 == features[2][2]
    
    # Test that peak finding in a partially overlapping range works
    assert 17 == features[0][3]
    assert 16 == features[1][3]
    assert 19 == features[2][3]
    
    # Test that finding the ratio of the peak over average values works
    assert 2.0 / 0.8 == features[0][4]
    assert 3.0 / 1.2 == features[1][4]
    assert 5.0 / 2.4 == features[2][4]

 
def test_label(default_data):
    '''
    Test getting label
    '''  
    
    filter = Filter.Filter()
    
    # Test finding a property in the data
    filter.labelPath = ["properties", "phase"]
    labels = filter.getLabels(default_data)
    assert ["solid", "solid", "liquid"] == labels
    
    # Test finding presense/absense of an item in a list
    filter.labelPath = ["components", "SSM:PRESENT:name:foo"]
    labels = filter.getLabels(default_data)
    assert ["foo_present", "foo_present", "foo_absent"] == labels
    
