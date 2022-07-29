import io
import pickle
import psycopg2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

def load_extrema(name, host, port, database_name, user, password):
    '''
    Retrieve a feature extrema list from the database
    
    @param name The name of the model whose extrema are to be retrieved.
    @param host The hostname of the Postgres database
    @param port The port number for the database
    @param database_name The name of the Postgres database
    @param user The database username
    @param password The dataabse password
    @return String with a comma separeted list of "feature 1 min, feature 1 max, feature 2 min..."
    '''
    
    # Connect to the database
    connection = psycopg2.connect(database=database_name, user=user, password=password, host=host)
    
    # Get the specified model
    cursor = connection.cursor()
    cursor.execute("SELECT extrema FROM models WHERE name = '" + name + "'")
    result = cursor.fetchall()
    
    if len(result) > 0:
        return result[0][0]
    else:
        return None
    
def load_filter(name, host, port, database_name, user, password):
    '''
    Retrieve a json description of a Filter from the database.
    
    @param name The name of the model whose Filter is to be retrieved.
    @param host The hostname of the Postgres database
    @param port The port number for the database
    @param database_name The name of the Postgres database
    @param user The database username
    @param password The dataabse password
    @return The Filter for the model from the database with the specified name, or None if nothing was found
    '''
    
    #Connect to the database
    connection = psycopg2.connect(database = database_name, user = user, password = password, host = host)
    
    #Get the specified model
    cursor = connection.cursor()
    cursor.execute("SELECT filter FROM models WHERE name = '" + name + "'")
    result = cursor.fetchall()
    
    if len(result) > 0:
        return result[0][0]
    else:
        return None
    
def load_labels(name, host, port, database_name, user, password):
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
    cursor.execute("SELECT labels FROM models WHERE name = '" + name + "'")
    result = cursor.fetchall()
    
    if len(result) > 0:
        return result[0][0]
    else:
        return None

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

def predict(filters, data, model, host, port, database_name, user, password):
    '''
    Predict the class of the given data, using the named model.
    
    @param filters List of Filters to apply to the data, which must match those used to train the model
    @param data Dictionary representation of the data point's JSON.
    @param model The name of the model to use
    @param host: Hostname where the database is located.
    @param port: The port where the database is available as a string
    @param database_name: The name of the database 
    @param user: The username for the database
    @param password: The password for the database
    '''
    
    # Get the features for the current data point
    for filter in filters:
        features = filter.getFeatures(data)
    
    # Get the extrema from the database and convert them into numbers
    extrema_string = load_extrema(model, host, port, database_name, user, password)
    
    extrema = []
    
    for number in extrema_string.split(","):
        extrema.append(float(number))
    
    # For each feature, convert it into a number in range [0,1] instead of the range [min, max]
    for i in range(len(features)):
        features[i][0] = (features[i][0] - extrema[i * 4]) / (extrema[i * 4 + 1] - extrema[i * 4])
        features[i][1] = (features[i][1] - extrema[i * 4 + 2]) / (extrema[i * 4 + 3] - extrema[i * 4 + 2])
        
    # Load the classifier from the database
    classifier = load_model(model, host, port, database_name, user, password)
    
    # Load the class labels from the database
    labels = load_labels(model, host, port, database_name, user, password).split(",")
    
    # Get the probability
    if hasattr(classifier, "decision_function"):
        prob = classifier.decision_function([features[0]])
    else:
        prob = classifier.predict_proba([features[0]])[:, 0]
        
    return labels, prob

def save_model(classifier, name, training_data, filter_json, feature_extrema, labels, description, host, port, database_name, user, password, filter, features, data_labels):
    '''
    Serialize the given model and save it to a Postgres database.
    
    @param classifier: The model to be saved.
    @param name: The name to save the classifier under
    @param training_data: The list of datasets used in training the model
    @param filter: The json representation (as used by construct_filters) of the filter used to extract features/labels.
    @param feature_extrema: The list of extrema for each feature in the form [[f1_min, f1_max], [f2_min, f2_max]...]
    @param labels: The ordered list of labels for the classifier
    @param description: A string containing a human readable explanation of what the model is and how it works.
    @param host: Hostname where the database is located.
    @param port: The port where the database is available as a string
    @param database_name: The name of the database 
    @param user: The username for the database
    @param password: The password for the database
    '''
    
    #Create a grid to graph
    x_min, x_max = 0,1.1 #-500, 5000
    y_min, y_max = 0,1.1 #20000, 55000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                              np.arange(y_min, y_max, .05))
    
    # The grid, in the correct ranges rather than normalized to the range [0,1]
    full_xx, full_yy = np.meshgrid(np.arange(x_min, x_max, .05),
                              np.arange(y_min, y_max, .05))
    
    #Calculate the x coordinates of the grid
    for j in full_xx:
        for l, k in enumerate(j):
            j[l] = filter.feature_mins[0] + j[l] * (filter.feature_maxs[0] - filter.feature_mins[0])
            
    #Calculate the y coordinates of the grid, or if there is only one feature leave it
    for j in full_yy:
        for l, k in enumerate(j):
            if len(filter.featurePaths) == 1:
                j[l] = j[l] 
            else:
                j[l] = filter.feature_mins[1] + j[l] * (filter.feature_maxs[1] - filter.feature_mins[1])

                
    if len(filter.featurePaths) == 1:
    
        #Get the probability function over the grid
        if hasattr(classifier, "decision_function"):
            zz = classifier.decision_function(np.c_[xx.ravel()])
        else:
            zz = classifier.predict_proba(np.c_[xx.ravel()])[:, 1]
    
    else:

        #Get the probability function over the grid
        if hasattr(classifier, "decision_function"):
            zz = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            zz = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    #Plot the background probability
    zz = zz.reshape(xx.shape)
    plt.contourf(full_xx, full_yy, zz, cmap=plt.cm.RdBu, alpha=.8)

    if len(filter.featurePaths) == 1:
        plt.scatter([filter.feature_mins[0] + (i[0] * (filter.feature_maxs[0] - filter.feature_mins[0])) for i in features], [0 for i in features], c=['blue' if x == classifier.classes_[1] else 'red' for x in data_labels], cmap=ListedColormap(['#FF0000', '#0000FF']), alpha=0.6,
                edgecolors='k')    
    else:
        plt.scatter([filter.feature_mins[0] + (i[0] * (filter.feature_maxs[0] - filter.feature_mins[0])) for i in features], [filter.feature_mins[1] + (i[1] * (filter.feature_maxs[1] - filter.feature_mins[1])) for i in features], c=['blue' if x == classifier.classes_[1] else 'red' for x in data_labels], cmap=ListedColormap(['#FF0000', '#0000FF']), alpha=0.6,
                edgecolors='k')  
    
    
    #Create the title based on the label
    for path in filter.labelPath:
        if "SSM:PRESENT" in path:
            tokens = path.split(":")
            plt.title(tokens[2] + " = " + tokens[3])
    
    # Set the x axis description 
    for path in filter.featurePaths[0]:
        if isinstance(path, str):
            if "SSM:" in path:
                if ":PEAK-DISTANCE:" in path:
                    tokens = path.split(":")
                    plt.xlabel("Whether " + tokens[8] + " peaks in range " + tokens[4] + "-" + tokens[5] + " spaced " + str(int(tokens[6]) - int(tokens[7])) + "-" + str(int(tokens[6]) + int(tokens[7])) + " of min width " + tokens[9])
                    break
                elif ":PEAK-LOC" in path:
                    plt.xlabel("Peak Position")
                    break
                elif ":PEAK-RATIO-RANGE:" in path:
                    tokens = path.split(":")
                    plt.xlabel("Ratio of highest/lowest values in range " + tokens[4] + "-" + tokens[5])
                    break
                elif ":PEAKS-RATIO-TWO-RANGES:" in path:
                    tokens = path.split(":")
                    plt.xlabel("Ratio of peak heights for peaks in " + tokens[4] + "-" + tokens[5] + " and " + tokens[6] + "-" + tokens[7])
                    break

    if (len(filter.featurePaths) > 1):
        for path in filter.featurePaths[1]:
            if isinstance(path, str):
                if "SSM:" in path:
                    if ":PEAK-DISTANCE:" in path:
                        tokens = path.split(":")
                        plt.ylabel("Whether " + tokens[8] + " peaks in range " + tokens[4] + "-" + tokens[5] + " spaced " + str(int(tokens[6]) - int(tokens[7])) + "-" + str(int(tokens[6]) + int(tokens[7])) + " of min width " + tokens[9])
                        break
                    elif ":PEAK-LOC" in path:
                        plt.ylabel("Peak Position")
                        break
                    elif ":PEAK-RATIO-RANGE:" in path:
                        tokens = path.split(":")
                        plt.ylabel("Ratio of highest/lowest values in range " + tokens[4] + "-" + tokens[5])
                        break
                    elif ":PEAKS-RATIO-TWO-RANGES:" in path:
                        tokens = path.split(":")
                        plt.ylabel("Ratio of peak heights for peaks in " + tokens[4] + "-" + tokens[5] + " and " + tokens[6] + "-" + tokens[7])
                        break
    
    #Get the binary image data for the graph
    blob = io.BytesIO()
    plt.savefig(blob, format='png')
    blob.seek(0)
    
    # Commit the graph to the database
    connection = psycopg2.connect(database=database_name, user=user, password=password, host=host, port=port)
    connection.cursor().execute("INSERT INTO graphs (image, model) VALUES (%s, %s)", (blob.read(), name))
    connection.commit()
    
    #Get the label for each training data point
    if hasattr(classifier, "decision_function"):
        # foo = np.c_[[feature_mins[0] + (i[0] * (feature_maxs[0] - feature_mins[0])) for i in features],[feature_mins[1] + (i[1] * (feature_maxs[1] - feature_mins[1])) for i in features]]
        foo = np.c_[features]
        foo = np.c_[[1000, 2000, 3000, 4000, 5000], [0, 0, 0, 0, 0]]
        # #print(xx.ravel())
        # print(zz)
        # print(clf.predict(foo))
        # predictions = clf.predict(foo)
        predictions = data_labels
    else:
        foo = np.c_[features]
        predictions = classifier.predict(foo)
    
    #Caclulate True/False negatives/positives
    tp = 0
    tn = 0
    fp = 0
    fn = 0
        
    for i, label in enumerate(data_labels):
        if label.endswith("_absent"):
            if predictions[i].endswith("_absent"):
                tn += 1
            else:
                fp += 1
        else:
            if predictions[i].endswith("_absent"):
                fn += 1
            else:
                tp += 1
                
    # Calculate metrics
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    
    if tp + fn > 0:
        recall = (1.0 * tp) / (tp + fn)
    else:
        recall = 0
    if fp + tp > 0:
        false_discovery = (1.0 * fp) / (fp + tp)
    else:
        false_discovery = (1.0 * fp) / (fp + tp)
    if tn + fp > 0:
        selectivity = (1.0 * tn) / (tn + fp)
    else:
        selectivity = 0
    if fn + tn > 0:
        false_omission = (1.0 * fn) / (fn + tn)
    else:
        false_omission = 0
    
    #Pickle the classifier
    model_pickle = psycopg2.Binary(pickle.dumps(classifier))
    
    #Get the list of dataset uuids
    datasets = ''
    #for dataset in training_data:
    #    datasets += dataset['uuid']
    
    extrema_string = ''
    
    for ex in feature_extrema:
        extrema_string += str(ex[0]) + ',' + str(ex[1]) + ','
        
    extrema_string = extrema_string[:-1]
    
    label_string = ''
    
    for label in classifier.classes_:
        label_string += label + ","
        
    label_string = label_string[:-1]

    
    connection.cursor().execute("INSERT INTO models (name, datasets, model, filter, extrema, labels, description, f1, recall, selectivity, false_discovery, false_omission, true_positive, true_negative, false_positive, false_negative) VALUES ('" + name + "', '" 
                                + datasets + "', " + str(model_pickle) + ", '" + filter_json + "', '" + extrema_string + "', '"
                                + label_string + "', '" + description 
                                + "'," + str(f1) + ", " 
                                +str(recall) + ", " + str(selectivity) + ", " + str(false_discovery) + ", " + str(false_omission) + ", " + str(tp) + ", "
                                +str(tn) + ", " + str(fp) + ", " + str(fn) 
                                +")")
    
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
    