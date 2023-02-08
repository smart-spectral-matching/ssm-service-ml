import pytest 

from ssm_ml import Filter
from ssm_ml import ssmml


def test_construct_filters(): 
    '''
    Test constructing filters from json
    '''
    json_filters =     {
        "Filters" : [
            {
                "Features" : [ 
                    ["path1", "path2", "path3"],
                    ["path4", "path5", "path6"],
                ],
                "Label" : ["path7", "path8", "path9"]
            },
            {
                "Features" : [ 
                    ["path10", "path11", "path12"],
                    ["path13", "path14", "path15"],
                ],
                "Label" : ["path16", "path17", "path18"]
            },
        ]
    }
        
    #Create the filters
    filters = ssmml.construct_filters(json_filters)
    
    #Check that the filters match the json
    assert 2 == len(filters)
    assert ["path1", "path2", "path3"] == filters[0].featurePaths[0]
    assert ["path4", "path5", "path6"] == filters[0].featurePaths[1]
    assert ['path7','path8','path9'] == filters[0].labelPath
    assert ["path10", "path11", "path12"] == filters[1].featurePaths[0]
    assert ["path13", "path14", "path15"] == filters[1].featurePaths[1]
    assert ['path16','path17','path18'] == filters[1].labelPath

 
def test_database():
    '''
    Test saving and loading a model with the database
    '''
    #Create a simple classifier
    features = [[0,0],[1,1]]
    labels = ['off', 'on']
    classifier = ssmml.train(labels, features, "SVC")
    filter_json = '{ "foo" : 4, "bar" : "none" }'
    filter = Filter.Filter()
    filter.feature_mins = [0, 0]
    filter.feature_maxs = [1, 1]
    filter.featurePaths=[[],[]]
     
    ssmml.save_model(
        classifier,
        "test",
        ['0', '1'],
        filter_json,
        [[0,1],[0,1]],
        labels,
        "sample model for testing",
        "172.17.0.2",
        "5432",
        "ssm",
        "postgres",
        "postgres",
        filter,
        [[0,0], [1,1]],
        ["off", "on"]
    )

    db_classifier = ssmml.load_model(
        'test',
        "172.17.0.2",
        "5432",
        "ssm",
        "postgres",
        "postgres"
    )
     
     
    assert -0.9 > db_classifier.decision_function([[0.1,0.1]])
     
    db_filter = ssmml.load_filter(
        'test',
        "172.17.0.2",
        "5432",
        "ssm",
        "postgres",
        "postgres"
    )
    assert filter_json == db_filter

 
def test_normalize_features():
    '''
    Test feature normalization.
    '''
    features = [[0, 1, 5.5], [1, 4, 3.5], [2, 16, 2]]
    
    features = ssmml.normalize_features(features)
    
    #Check that the minima are 0s, the maximas are 1s, and the middle values are properly porportnate
    assert [0, 0, 1] == features[0]
    assert [0.5, 0.2, 0.42857142857142855] == features[1]
    assert [1, 1, 0] == features[2]

 
def test_train():
    '''
    Test the training of a model
    '''
    #Create a simple classifier
    features = [[0,0],[1,1]]
    labels = ['off', 'on']
    classifier = ssmml.train(labels, features, "SVC")
    
    #Check that it works
    classifier.decision_function([[0.1,0.1]])
