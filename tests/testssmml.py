import unittest

from ssm_ml import ssmml

class TestSsmml(unittest.TestCase):
    '''
    Test class for the ssmml library 
    
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [0, 1, 2, 4, 8, 16, 8, 4, 2, 1]
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [0, 1, 2, 4, 15, 7, 8, 4, 2, 1]
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [0, 1, 2, 4, 8, 10, 12, 14, 16, 18]
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 1, 2, 1, 0]
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 3, 2, 1, 0 ]
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
                                "axis" : "x-axis",
                                "data" : [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                                },
                            {
                                "axis" : "y-axis",
                                "data" : [21, 22, 24, 22, 21, 0, 4, 1, 2, 5]
                                }
                    ]
                }
            ]
    
    def test_construct_filters(self): 
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
        self.assertEqual(2, len(filters))
        self.assertListEqual(["path1", "path2", "path3"], filters[0].featurePaths[0])
        self.assertListEqual(["path4", "path5", "path6"], filters[0].featurePaths[1])
        self.assertListEqual(['path7','path8','path9'], filters[0].labelPath)
        self.assertListEqual(["path10", "path11", "path12"], filters[1].featurePaths[0])
        self.assertListEqual(["path13", "path14", "path15"], filters[1].featurePaths[1])
        self.assertListEqual(['path16','path17','path18'], filters[1].labelPath)
        
    def test_database(self):
        '''
        Test saving and loading a model with the database
        '''
        
        #Create a simple classifier
        features = [[0,0],[1,1]]
        labels = ['off', 'on']
        classifier = ssmml.train(labels, features, "SVC")
        
        ssmml.save_model(classifier, "test", ['0', '1'], "172.17.0.2", "5432", "ssm", "postgres", "postgres")
        db_classifier = ssmml.load_model('test', "172.17.0.2", "5432", "ssm", "postgres", "postgres")
        
        self.assertTrue(-0.9 > db_classifier.decision_function([[0.1,0.1]]))
        
    def test_normalize_features(self):
        '''
        Test feature normalization.
        '''
        
        features = [[0, 1, 5.5], [1, 4, 3.5], [2, 16, 2]]
        
        features = ssmml.normalize_features(features)
        
        #Check that the minima are 0s, the maximas are 1s, and the middle values are properly porportnate
        self.assertListEqual([0, 0, 1], features[0])
        self.assertListEqual([0.5, 0.2, 0.42857142857142855], features[1])
        self.assertListEqual([1, 1, 0], features[2])
        
    def test_train(self):
        '''
        Test the training of a model
        '''
        
        #Create a simple classifier
        features = [[0,0],[1,1]]
        labels = ['off', 'on']
        classifier = ssmml.train(labels, features, "SVC")
        
        #Check that it works
        classifier.decision_function([[0.1,0.1]])