import copy
import io
import math
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

def interpolate_spectra(spectra1, spectra2):
    '''
    Create a copy of spectra2 such that the x values are the same as those for spectra1, using interpolation to estimate
    values.

    @param spectra1 A list of lists of spectra values to serve as the prototype.
    @param spectra2 A list of lists of spectra values to cast into the same wavenumbers as spectra1.
    @return A list of lists of spectra values, such that the first list (the x axis) is equal to the first list in spectra1
        and the second list (the y axis) contains estimated values of spectra2 at those wave numbers.
    '''

    # The new spectra will have the same x axis as spectra1
    new_spectra = [copy.deepcopy(spectra1[0]),[]]

    # Last checked index in spectra 2
    j = 0

    for i in range(len(spectra1[0])):
        target_x = spectra1[0][i]

        # If spectra2 extends beyond the bounds of spectra1, use the endpoint value
        if spectra2[0][0] > target_x:
            new_spectra[1].append(spectra2[1][0])
        elif spectra2[0][-1] < target_x:
            new_spectra[1].append(spectra2[1][-1])
        else:
            # Search for the target value in spectra2's x axis
            while j < len(spectra2[0]):
                candidate_x = spectra2[0][j]

                # If the exact value exists, use its y data
                if target_x == candidate_x:
                    new_spectra[1].append(spectra2[1][j])
                    break

                # We skipped past the target value, so interpolate
                elif target_x < candidate_x:

                    # Get the x/y values to the left and right of the target x value
                    left_x = spectra2[0][j - 1]
                    left_y = spectra2[1][j - 1]
                    right_x = spectra2[0][j]
                    right_y = spectra2[1][j]

                    # Horizontal and vertical distance between left/right points
                    distance = right_x - left_x
                    slope = right_y - left_y

                    # Add the proportion of the y distance equal to the proportion of x distance from the end to the target
                    # This is the intersection between a perpindicular line from target_x to the line between the left and
                    # right points
                    interpolated_y = left_y + (float(target_x - left_x) / distance * slope)

                    new_spectra[1].append(interpolated_y)
                    break

                j = j + 1

    return new_spectra

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

def match_spectra(search_spectra, datasets):
    '''
    Find the spectra from the database that match the given spectra.

    @param search_spectra List of two lists of floats, being the x and y axes, respectively of the spectra to match.
    @param datasets List of all datasets to match against, in SSM dictionary format.
    '''

    max_pcc = 0
    max_pcc_name = ""
    max_sec = 0
    max_sec_name = ""
    max_sfec = 0
    max_sfec_name = ""
    max_uned = 0
    max_uned_name = ""
    max_all = 0
    max_all_name = ""

    for ds in datasets:
        candidate = [[], []]
        candidate[0] = ds['scidata']['dataseries'][0]["x-axis"]['parameter']['numericValueArray'][0]['numberArray']
        candidate[1] = ds['scidata']['dataseries'][1]["y-axis"]['parameter']['numericValueArray'][0]['numberArray']
        s1, s2 = truncate_spectra(search_spectra, candidate)
        i1, i2 = interpolate_spectra(s1, s2)

        s1[1] = i1
        s2[1] = i2

        pcc = pearson_correlation_coefficient(s1, s2)
        if pcc > max_pcc:
            max_pcc = pcc
            max_pcc_name = ds['title']

        sfec = squared_first_difference_euclidean_cosine(s1, s2)
        if sfec > max_sfec:
            max_sfec = sfec
            max_sfec_name = ds['title']

        sec = squared_euclidean_cosine(s1, s2)
        if sec > max_sec:
            max_sec = sec
            max_sec_name = ds['title']

        uned = unit_normalized_euclidean_distance(s1, s2)
        if uned > max_uned:
            max_uned = uned
            max_uned_name = ds['title']

        allv = pcc + sfec + sec + (uned / 329696240.4570517)
        if allv > max_all:
            max_all = allv
            max_all_name = ds['title']

    print("Pearson Correlation Coefficient:")
    print(max_pcc)
    print(max_pcc_name)

    print("Squared Euclidean Cosine:")
    print(max_sec)
    print(max_sec_name)

    print("Squared First Order Euclidean Cosine:")
    print(max_sfec)
    print(max_sfec_name)

    print("Unit Normalized Eculidean Distance:")
    print(max_uned)
    print(max_uned_name)

    print("Overall:")
    print(max_all)
    print(max_all_name)

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

def pearson_correlation_coefficient(spectra1, spectra2):
    '''
    Calculate Pearson's correlation coefficient between the two spectra of the same length.

    @param spectra1 A list of lists of float spectra data, with one list for x and another for y
    @param spectra2 A list of lists of float spectra data, with one list for x and another for y
    @return The Pearson Correlation Coefficient as a float
    '''

    sum1 = sum(spectra1[1])
    sum2 = sum(spectra2[1])
    sum_square1 = sum([i ** 2 for i in spectra1[1]])
    sum_square2 = sum([i ** 2 for i in spectra2[1]])
    series_mult = []

    for i in range(len(spectra1[0])):
        series_mult.append(spectra1[1][i] * spectra2[1][i])

    sum_mult = sum(series_mult)

    numerator = sum1 * sum2 / len(spectra1[0])
    numerator = sum_mult - numerator

    denominator_term1 = sum1 ** 2 / len(spectra1[0])
    denominator_term1 = sum_square1 - denominator_term1
    denominator_term2 = sum1 ** 2 / len(spectra2[0])
    denominator_term2 = sum_square2 - denominator_term2
    denominator = denominator_term1 * denominator_term2

    denominator = math.sqrt(denominator)

    return numerator / denominator

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
    @param training_data: The list of collections used in training the model
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

    if 2.0 * tp + fp + fn > 0:
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    else:
        f1 = 0

    if tp + fn > 0:
        recall = (1.0 * tp) / (tp + fn)
    else:
        recall = 0

    if fp + tp > 0:
        false_discovery = (1.0 * fp) / (fp + tp)
    else:
        false_discovery = 0

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

    #Get the list of collection uuids
    collections = ''
    #for collection in training_data:
    #    collections += collection['uuid']

    extrema_string = ''

    for ex in feature_extrema:
        extrema_string += str(ex[0]) + ',' + str(ex[1]) + ','

    extrema_string = extrema_string[:-1]

    label_string = ''

    for label in classifier.classes_:
        label_string += label + ","

    label_string = label_string[:-1]


    connection.cursor().execute("INSERT INTO models (name, collections, model, filter, extrema, labels, description, f1, recall, selectivity, false_discovery, false_omission, true_positive, true_negative, false_positive, false_negative) VALUES ('" + name + "', '"
                                + collections + "', " + str(model_pickle) + ", '" + filter_json + "', '" + extrema_string + "', '"
                                + label_string + "', '" + description
                                + "'," + str(f1) + ", "
                                +str(recall) + ", " + str(selectivity) + ", " + str(false_discovery) + ", " + str(false_omission) + ", " + str(tp) + ", "
                                +str(tn) + ", " + str(fp) + ", " + str(fn)
                                +")")

    connection.commit()

    connection.close()

def squared_euclidean_cosine(spectra1, spectra2):
    '''
    Calculate the squared Euclidean cosine between the two spectra.

    @param spectra1 A list of lists of float spectra data, with one list for x and another for y
    @param spectra2 A list of lists of float spectra data, with one list for x and another for y
    @return The squared Euclidean cosine as a float
    '''

    array1 = np.array(spectra1[1])
    array2 = np.array(spectra2[1])

    numerator = np.sum(np.multiply(array1, array2)) ** 2

    square1 = np.square(array1)
    square2 = np.square(array2)

    sum1 = np.sum(square1)
    sum2 = np.sum(square2)

    denominator = sum1 * sum2

    if denominator == 0:
        return 0

    return numerator / denominator

def squared_first_difference_euclidean_cosine(spectra1, spectra2):
    '''
    Calculate the squared first-difference Euclidean cosine between the two spectra.

    @param spectra1 A list of lists of float spectra data, with one list for x and another for y
    @param spectra2 A list of lists of float spectra data, with one list for x and another for y
    @return The squared first-difference Euclidean cosine as a float
    '''

    array1 = np.array(spectra1[1])
    array1 = np.ediff1d(array1)
    array2 = np.array(spectra2[1])
    array2 = np.ediff1d(array2)

    numerator = np.sum(np.multiply(array1, array2)) ** 2

    square1 = np.square(array1)
    square2 = np.square(array2)

    sum1 = np.sum(square1)
    sum2 = np.sum(square2)

    denominator = sum1 * sum2

    if denominator == 0:
        return 0

    return numerator / denominator

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

def truncate_spectra(spectra1, spectra2):
    '''
    Truncate the spectra such that the two are guaranteed to have the same range. That is, if s1 ranges from 0 to 100 and
    s2 ranges from 5 to 150, the new spectra will range from 50 to 100.

    @param spectra1 A list of lists for x and y values of one spectra to truncate
    @param spectra2 A list of lists for x and y values of the other spectra to truncate
    @return Two lists of lists whose x axes start at max(lowest x in s1, lowest x in s2) and end at min(highest x in s1,
        highest x in s2) and are otherwise identical to s1 and s2 respectively.
    '''

    # Copy the spectra
    s1 = copy.deepcopy(spectra1)
    s2 = copy.deepcopy(spectra2)

    # Get the starting point of each
    start1 = s1[0][0]
    start2 = s2[0][0]

    # If they don't start at the same value, cut from the start
    if start1 != start2:

        cut_index = 0

        # If s1 starts first, truncate it
        if start1 < start2:

            # Find the cutoff where s1 reaches s2
            while s1[0][cut_index] < start2:
                cut_index += 1

            s1[0] = s1[0][cut_index:]
            s1[1] = s1[1][cut_index:]

        else:

            # Find the cutoff where s2 reaches s1
            while s2[0][cut_index] < start1:
                cut_index += 1

            s2[0] = s2[0][cut_index:]
            s2[1] = s2[1][cut_index:]

    # Get the ending point of each
    end1 = s1[0][-1]
    end2 = s2[0][-1]

    # If they don't end at the same value, cut from the end
    if end1 != end2:

        # If s2 ends last, truncate it
        if end1 < end2:

            cut_index = len(s2[0]) - 1

            # Find the cutoff where s2 reaches s1
            while s2[0][cut_index] > end1:
                cut_index -= 1

            s2[0] = s2[0][0:cut_index]
            s2[1] = s2[1][0:cut_index]

        else:

            cut_index = len(s1[0]) - 1

            # Find the cutoff where s1 reaches s2
            while s1[0][cut_index] > end1:
                cut_index -= 1

            s1[0] = s1[0][0:cut_index]
            s1[1] = s1[1][0:cut_index]

    return s1, s2

def unit_normalized_euclidean_distance(spectra1, spectra2):
    '''
    Calculate the unit normalized Eculidean distance between the two spectra.

    @param spectra1 A list of lists of float spectra data, with one list for x and another for y
    @param spectra2 A list of lists of float spectra data, with one list for x and another for y
    @return The unit normalized Eculidean distance as a float
    '''

    array1 = np.array(spectra1[1])
    array2 = np.array(spectra2[1])

    difference = np.subtract(array1, array2)
    value = np.square(difference)
    sum = np.sum(value)

    return math.sqrt(sum)

