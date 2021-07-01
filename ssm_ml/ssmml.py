import pickle
import psycopg2

from pip._vendor.distlib import database
from ssm_ml import Filter

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def construct_filters(json):
    '''
    Create a list of Filters from a JSON description.
    
    Filter descriptions should be of the form
    
    {
        "Filters" : [
            {
                "Features" : [ 
                    ["path1", "path2", "path3"...],
                    ["path1", "path2", "path3"...],
                    ...
                ]
                "Label" : ["path1", "path2", "path3"...]
            },
            {
                "Features" : [ 
                    ["path1", "path2", "path3"...],
                    ["path1", "path2", "path3"...],
                    ...
                ]
                "Label" : ["path1", "path2", "path3"...]
            },
            ...
        ]
    
    }
    
    @param json The json description containing the filters. 
    @return A list of Filters as described by the JSON
    '''
    
    #Filters under construction
    filters = []
    
    #List of filter descriptions
    filter_descs = json["Filters"]
    
    #Convert each description into a filter
    for filter_desc in filter_descs:
        
        filter = Filter.Filter()
        filter.labelPath = filter_desc["Label"]
        filter.featurePaths = filter_desc["Features"]
        filters.append(filter)
        
    return filters;

def load_model(name, host, port, database_name, user, password):
    '''
    Retrieve a pre-trained classifier model from the database.
    
    @param name The name of the model to retrieve.
    @param host The hostname of the Postgres database
    @param port The port number for the database
    @param database_name The name of the Postgres database
    @param user The database username
    @param password The dataabse password
    @return The model from the database with the specified name, or None if nothing was found
    '''
    
    #Connect to the database
    connection = psycopg2.connect(database = database_name, user = user, password = password, host = host)
    
    #Get the specified model
    cursor = connection.cursor()
    cursor.execute("SELECT model FROM models WHERE name = '" + name + "'")
    result = cursor.fetchall()
    
    if len(result) > 0:
        return pickle.loads(result[0][0])
    else:
        return None

def normalize_features(features):
    '''
    Normalize each feature into the range 0-1, 
    
    @param features The list of features to be normalized
    @return The list of features, normalized into the rang 0-1
    '''
    
    #The minima for each feature
    feature_mins = []
    
    #THe maxima for each feature
    feature_maxs = []
    
    #Initialize the extrema
    for f in features[0]:
        feature_mins.append(f)
        feature_maxs.append(f)
        
    #Find the extrema for all features
    for feature in features:
        for i in range(len(feature)):
            if feature[i] < feature_mins[i]:
                feature_mins[i] = feature[i]
            elif feature[i] > feature_maxs[i]:
                feature_maxs[i] = feature[i]
                
    #Normalize all features into the range 0-1
    for feature in features:
        for i in range(len(feature)):
            feature[i] = (feature[i] - feature_mins[i]) / (feature_maxs[i] - feature_mins[i]) 
            
    return features

def save_model(classifier, name, training_data, host, port, database_name, user, password):
    '''
    Serialize the given model and save it to a Postgres database.
    
    @param classifier: The model to be saved.
    @param name: The name to save the classifier under
    @param training_data: The list of datasets used in training the model
    @param host: Hostname where the database is located.
    @param port: The port where the database is available as a string
    @param database_name: The name of the database 
    @param user: The username for the database
    @param password: The password for the database
    '''
    
    #Pickle the classifier
    model_pickle = psycopg2.Binary(pickle.dumps(classifier))
    
    #Get the list of dataset uuids
    datasets = ''
    #for dataset in training_data:
    #    datasets += dataset['uuid']
    
    #Commit the new model to the database
    connection = psycopg2.connect(database = database_name, user = user, password = password, host = host)
    
    connection.cursor().execute("INSERT INTO models (name, datasets, model) VALUES ('" + name + "', '" + datasets + "', " + str(model_pickle) + ") ON CONFLICT (name) DO UPDATE SET datasets = EXCLUDED.datasets, model = EXCLUDED.model;")
    connection.commit()
    connection.close()

def train(labels, features, type):
    '''
    Train a classifier on the given data.
    
    The JSON type must be of the form 
    
    {
        "Classifier" : {
            "Type" : "type"
        }
    }
    
    Where type must be one of:
    
    KNeighborsClassifier
    SVC
    GaussianProcessClassifier
    DecisionTreeClassifier
    RandomForestClassifier
    MLPClassifier
    AdaBoostClassifier
    GaussianNB
    QuadraticDiscriminantAnalysis
    
    @param labels List of labels for each data point.
    @param features List of features for each data point.
    @param type JSON description of the classifier to create
    @return The trained classifier
    '''
    
    classifier = None
    
    #TODO Decide which parameters to open up to customization and parse them from the json
    #Create the specified classifier
    if type == "KNeighborsClassifier":
        classifier = KNeighborsClassifier(3)
    if type == "SVC":
        classifier = SVC(gamma=2, C=1)
    if type == "GaussianProcessClassifier":
        classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    if type == "DecisionTreeClassifier":
        classifier = DecisionTreeClassifier(max_depth=5)
    if type == "RandomForestClassifier":
        classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    if type == "MLPClassifier":
        classifier = MLPClassifier(alpha=1, max_iter=1000)
    if type == "AdaBoostClassifier":
        classifier = AdaBoostClassifier()
    if type == "GaussianNB":
        classifier = GaussianNB()
    if type == "QuadraticDiscriminantAnalysis":
        classifier = QuadraticDiscriminantAnalysis()
        
    classifier.fit(features, labels)
    
    return classifier
    