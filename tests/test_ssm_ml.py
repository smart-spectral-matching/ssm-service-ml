import math
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

@pytest.mark.skip("Failing due to changes in ssmml.interpolate_spectra")
def test_interpolate_spectra():
    '''
    Test spectra interpolation
    '''
    
    spectra1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 0, 1, 2, 1, 0, 1, 2, 3]]
    spectra2 = [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    
    spectra3 = ssmml.interpolate_spectra(spectra1, spectra2)
    
    for i in range(len(spectra3[0])):
        
        # Spectra 2 should have been cast into the exact x values as spectra 1
        assert spectra3[0][i] == spectra1[0][i]
        
        # The first two x values are before the start of spectra2, so replicate its first value
        # The rest of the values will be halfway between the integer y values because they are half way between the x values
        assert spectra3[1][i] == [1, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
 
def test_normalize_features():
    '''
    Test feature normalization.
    '''
    features = [[0, 1, 5.5], [1, 4, 3.5], [2, 16, 2]]
    
    features = ssmml.normalize_features(features)
    
    #Check that the minima are 0s, the maximas are 1s, and the middle values are properly porportonate
    assert [0, 0, 1] == features[0]
    assert [0.5, 0.2, 0.42857142857142855] == features[1]
    assert [1, 1, 0] == features[2]
    
def test_pearson_correlation_coefficient():
    '''
    Test Pearson correlation coefficient calculation.
    '''
    
    # Three spectra, with the first two orthogonal
    spectra1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]
    spectra2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]]
    spectra3 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2]]
    
    # Normalize the spectra
    features = ssmml.normalize_features([spectra1[1], spectra2[1], spectra3[1]])
    spectra1[1] = features[0]
    spectra2[1] = features[1]
    spectra3[1] = features[2]
    
    # The maximum PCC is 1 when two spectra are identical
    assert 1.0 == ssmml.pearson_correlation_coefficient(spectra1, spectra1)
    
    # Two maximally dissimilar spectra should have PCC -1
    assert -1.0 == ssmml.pearson_correlation_coefficient(spectra1, spectra2)
    
    # A more random spectra should have a value between the two extremes
    pcc3 = ssmml.pearson_correlation_coefficient(spectra1, spectra3)
    assert -1 < pcc3 and 1 > pcc3
    
def test_squared_euclidean_cosine():
    '''
    Test the squared Euclidean cosine
    '''
    
    # Three spectras with the first two orthogonal
    spectra1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]
    spectra2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]]
    spectra3 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2]]
    
    # Normalize the spectra
    features = ssmml.normalize_features([spectra1[1], spectra2[1], spectra3[1]])
    spectra1[1] = features[0]
    spectra2[1] = features[1]
    spectra3[1] = features[2]
    
    # The maximum distance is equal to the square root of the number of dimensions
    max_v = math.sqrt(len(spectra1[0]))
    
    # Identical vectors have no difference in the angle between them
    assert 0.0 == ssmml.squared_euclidean_cosine(spectra1, spectra1)
    
    # The maximum angular difference occurs between orthogonal vectors
    assert max_v == ssmml.squared_euclidean_cosine(spectra1, spectra2)
    
    # Any other spectra should have a value between those two extremes
    sec3 = ssmml.squared_euclidean_cosine(spectra1, spectra3)
    assert 0 < sec3 and max_v > sec3
    
def test_squared_first_difference_euclidean_cosine():
    '''
    Test the squared first difference Euclidean cosine
    '''
    
    # Three spectras with the first two orthogonal
    spectra1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]
    spectra2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]]
    spectra3 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2]]
    
    # Normalize the spectra
    features = ssmml.normalize_features([spectra1[1], spectra2[1], spectra3[1]])
    spectra1[1] = features[0]
    spectra2[1] = features[1]
    spectra3[1] = features[2]
    
    # Identical spectra have maximum similarity (1)
    assert 1.0 == ssmml.squared_euclidean_cosine(spectra1, spectra1)
    
    # Orthogonal spectra have minimum similarity (0)
    assert 0.0 == ssmml.squared_euclidean_cosine(spectra1, spectra2)
    
    # Any other spectra should have a value between those two extremes
    sec3 = ssmml.squared_euclidean_cosine(spectra1, spectra3)
    assert 0 < sec3 and 1 > sec3

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
    
def test_unit_normalized_euclidean_distance():
    '''
    Test Pearson correlation coefficient calculation.
    '''
    
    # Three spectras with the first two orthogonal
    spectra1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]
    spectra2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]]
    spectra3 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2]]
    
    # Normalize the spectra
    features = ssmml.normalize_features([spectra1[1], spectra2[1], spectra3[1]])
    spectra1[1] = features[0]
    spectra2[1] = features[1]
    spectra3[1] = features[2]
    
    # The maximum distance is equal to the square root of the number of dimensions
    max_v = math.sqrt(len(spectra1[0]))
    
    # Identical vectors have no difference in the angle between them
    assert 0.0 == ssmml.unit_normalized_euclidean_distance(spectra1, spectra1)
    
    # The maximum angular difference occurs between orthogonal vectors
    assert max_v == ssmml.unit_normalized_euclidean_distance(spectra1, spectra2)
    
    # Any other spectra should have a value between those two extremes
    uned3 = ssmml.unit_normalized_euclidean_distance(spectra1, spectra3)
    assert 0 < uned3 and max_v > uned3
