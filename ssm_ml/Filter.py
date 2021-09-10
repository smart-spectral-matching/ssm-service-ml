class Filter:
    '''
    A filter that finds the features and labels in datasets.
    
    '''
    
    def __init__(self):
        '''
        The default constructor
        '''
        self.labelPath = []
        self.featurePaths = []
        
    def getFeatures(self, dicts):
        '''
        Find the features for the given dictionaries.
        
        @param dicts: The dictionaries to extract the feature data for
        @return: A list of lists, where [i][j] corresponds to the jth feature defined in featurePaths for the 
            ith dictionary in dicts
        '''
        features = []
        
        # Find each defined feature
        for path in self.featurePaths:
            features.append(self._getNextSegment(dicts, path))
            
        final_features = []
        
        # Restructure the list so that each sub-list is for all features for a specific dictionary
        for i in range(len(features[0])):
            new_feature = []
            for feature in features:
                new_feature.append(feature[i])
            final_features.append(new_feature)
            
        return final_features
        
    def getLabels(self, dicts):
        '''
        Find the labels for the given dictionaries
        
        @return: A list of labels for each dictionary, in order
        '''
        return self._getNextSegment(dicts, self.labelPath)
        
    def _getNextSegment(self, dicts, path):
        '''
        Recursively traverse through the dictionaries one path segment at a time
        
        @param dicts: The dictionaries to traverse
        @param path: A list of ordered paths defining the data location in the dictionaries. Paths are either a literal key
            in the dictionary or one of:
            SSM:REG:foo The value of the fist key in the current list that matches regular expression "foo".
            SSM:XY:foo:PEAK The largest y value in the dictionary specified by the list item that has "foo" = "y-axis".
            SSM:XY:foo:PEAK-LOC The x value x[i] such that y[i] is maximal in the two list items where "foo" = "x-axis" 
                and "foo" = "y-axis" respectively.
            SSM:XY:foo:PEAK-LOC-RANGE:start:end The x value x[i] such that y[i] is maximal where start <= x[i] <= end in the
                two list items where "foo" = "x-axis" and "foo" = "y-axis" respectively.
            SSM:XY:foo:PEAK-RATIO-RANGE:start:end The ratio of a/b where a is the maximum and b the average for y[i] 
                considering only those values where start <= x[i] <= end for the two list items where "foo" = "x-axis" and
                "foo" = "y-axis" respectively.
            SSM:XY:foo:PEAKS-RATIO-TWO-RANGES:start1:end1:start2:end2 The ratio a/b where a is the maximum y[i] where start1 
                <= x[i] <end1 and b is the maximum y[i] where start2 <= x[i] <= end2
            -
        @return A list of the values for each dictionary in dicts at the path defined by path, in order. 
        '''
        
        next_values = []
        
        # SSM: is the flag for special actions in the path
        if path[0].startswith("SSM:"):
            
            # SSM:REG: defines selecting by a regular expression 
            if path[0].startswith("SSM:REG:"):
                tokens = path[0].split(":")
                
                # The key for the value to be matched
                key = tokens[2]
                
                # The regular expression to map
                reg = re.compile(tokens[3])
                
                # For each dictionary, find the sub-dictionaryy in the list such that candidate[key] matches
                # The regex
                for curr_dict in dicts:
                    for candidate in curr_dict:
                        if reg.match(candidate[key]):
                            next_values.append(candidate)
                            break;
            
            # SSM:XY: defines some calculation on XY data
            elif path[0].startswith("SSM:XY:"):
                tokens = path[0].split(":")
                
                # The key where the axis type is defined
                key = tokens[2]
                
                # The operation to be performed
                op = tokens[3]
                
                x_next = []
                y_next = []
                
                # For each dictionary, find the sub-dictionaries in the list for the x and y axes.
                for curr_dict in dicts:
                    
                    x_next.append(curr_dict[0]['x-axis'])
                    y_next.append(curr_dict[0]['y-axis'])
                    
#                     for candidate in curr_dict:
#                     
#                         if not key in candidate:
#                             continue
#                     
#                         x_found = False
#                         y_found = False
# 
#                         if candidate[key] == 'x-axis':
#                             x_next.append(candidate)
#                             x_found = True
#                         elif candidate[key] == 'y-axis':
#                             y_next.append(candidate)
#                             y_found = True
#                             
#                         if x_found and y_found:
#                             break;
                        
                # Continue the search down for the x and y data
                x = self._getNextSegment(x_next, path[1:])
                y = self._getNextSegment(y_next, path[1:])
                    
                # The PEAK operation finds the highest value in the data
                if op == "PEAK":
                    
                    peaks = []
                    
                    # Find the peak in each series
                    for series in y:
                        
                        peak = series[0]
                        
                        for i in series:
                            if i > peak:
                                peak = i
                                
                        peaks.append(peak)
                        
                    return peaks
                
                # The PEAK-LOC finds the values of x such that y[i] is the maximum y and x = x[i]
                elif op == "PEAK-LOC":
                    
                    peaks = []
                
                    # Find the location of the x where the maximum y is located
                    for i, series in enumerate(y):
                        
                        loc = 0
                        
                        for j in range(len(series)):
                            if series[j] > series[loc]:
                                loc = j
                                
                        peaks.append(x[i][loc])
                        
                    return peaks
                
                elif op.startswith("PEAK-LOC-RANGE"):
                    
                    range_tokens = op.split("-")
                    
                    start = float(range_tokens[3])
                    end = float(range_tokens[4])
                    
                    peaks = []
                    
                    ranges = []
                    
                    # Find the valid x locations in the range
                    for i, series in enumerate(x):
                        
                        # The valid endpoints that are in the range
                        curr_range = [0, 0]
                        
                        # Check whether the series starts in the range
                        if series[0] >= start:
                            in_range = True
                        else:
                            in_range = False
                        
                        '''
                        #Check each value in the series,
                        for j in range(len(series)):
                            
                            #Check for the end of the range
                            if in_range:
                                if series[j] >= end:
                                    
                                    #Go back one value if we skipped the start end
                                    if series[j] > end:
                                        curr_range[1] = j - 1
                                    else: 
                                        curr_range[1] = j
                                    
                                    ranges.append(curr_range)
                                    break
                            
                            else:
                                
                                #Check for the start of the range
                                if series[j] >= start:
                                    
                                    #Go back one value if we skipped the exact start
                                    if series[j] > start:
                                        curr_range[0] = j - 1
                                    else:
                                        curr_range[0] = j 
                                        
                        curr_range[1] = len(series) - 1
                        ranges.append(curr_range)
                    '''
                        
                    # Find the location of the x where the maximum y is located
                    for i, series in enumerate(y):
                        
                        loc = 0
                        min_loc = 0
                        
                        max = 0
                        min = 0
                        
                        for j in range(len(series)):
                            if x[i][j] >= start and x[i][j] <= end and series[j] > max:
                                loc = j
                                max = series[j]
                            # elif  x[i][j] >= start and x[i][j] <= end and series[j] < min:
                            #    min_loc = j
                            #    min = series[j]
                                
                        # if loc != min_loc:
                        #    peaks.append(series[loc] / series[min_loc])
                        # else:
                        #    peaks.append(0)
                        peaks.append(x[i][loc])
                        
                    return peaks
                
                # Find the ratio between the highest/lowest values in this range
                elif op.startswith("PEAK-RATIO-RANGE"):
                    
                    # Parse the tokens
                    range_tokens = op.split("-")
                    
                    start = float(range_tokens[3])
                    end = float(range_tokens[4])
                    
                    #
                    peaks = []
                    
                    ranges = []
                    
                    '''
                    #Find the valid x locations in the range
                    for i, series in enumerate(x):
                        
                        #The valid endpoints that are in the range
                        curr_range = [0,0]
                        
                        #Check whether the series starts in the range
                        if series[0] >= start:
                            in_range = True
                        else:
                            in_range = False
                        
                        #Check each value in the series,
                        for j in range(len(series)):
                            
                            #Check for the end of the range
                            if in_range:
                                if series[j] >= end:
                                    
                                    #Go back one value if we skipped the start end
                                    if series[j] > end:
                                        curr_range[1] = j - 1
                                    else: 
                                        curr_range[1] = j
                                    
                                    ranges.append(curr_range)
                                    break
                            
                            else:
                                
                                #Check for the start of the range
                                if series[j] >= start:
                                    
                                    #Go back one value if we skipped the exact start
                                    if series[j] > start:
                                        curr_range[0] = j - 1
                                    else:
                                        curr_range[0] = j 
                                        
                        curr_range[1] = len(series) - 1
                        ranges.append(curr_range)
                    '''
                        
                    # Find the location of the x where the maximum y is located
                    for i, series in enumerate(y):
                        
                        loc = 0
                        min_loc = 0
                        avg = 0
                        count = 0
                        max = 0
                        min = 9223372036854775807
                        
                        for j in range(len(series)):
                            if x[i][j] >= start and x[i][j] <= end:
                                avg = avg + series[j]
                                count = count + 1
                            if x[i][j] >= start and x[i][j] <= end and series[j] > max:
                                loc = j
                                max = series[j]
                            elif  x[i][j] >= start and x[i][j] <= end and series[j] < min:
                                min_loc = j
                                min = series[j]
                                
                        if loc != min_loc:
                            # peaks.append((sum(series[loc - 1:loc + 2]) / 3) / ( avg / count))
                            # if(sum(series[loc - 1:loc + 2]) / 4 - ( avg / count)):
                            #    print("" + str(series[loc]))
                            peaks.append(series[loc] / (avg / count))
                        else:
                            peaks.append(1)
                        # peaks.append(x[i][loc])
                        
                    return peaks
                
                # PEAKS-RATIO-TWO-RANGES means search within two ranges, and get the ratio of the peak in each range
                elif op.startswith("PEAKS-RATIO-TWO-RANGES"):
                    
                    # Parse the tokens
                    range_tokens = op.split("-")
                    
                    # Get the start and endpoint for each range
                    start1 = float(range_tokens[4])
                    end1 = float(range_tokens[5])
                    start2 = float(range_tokens[6])
                    end2 = float(range_tokens[7])
                    
                    ratios = []
                        
                    # Find max y in each of the two ranges
                    for i, series in enumerate(y):
                     
                        max1 = 0
                        max2 = 0 
                        
                        for j in range(len(series)):
                            if x[i][j] >= start1 and x[i][j] <= end1:
                                if series[j] > max1:
                                    max1 = series[j]
                            if x[i][j] >= start2 and x[i][j] <= end2:
                                if series[j] > max2:
                                    max2 = series[j]
                        
                        # Add the ratio between the two peaks
                        ratios.append(float(max1) / float(max2))
                        
                    return ratios
                
            # The SSM:PRESENT command tries to find a key-value pair and returns value_present or value_absent
            elif path[0].startswith("SSM:PRESENT"):
                tokens = path[0].split(":")
                key = tokens[2]
                value = tokens[3]
                
                # Such each sub dictionary for the key/value pair
                for curr_dict in dicts:
                    
                    found = False
                    
                    for candidate in curr_dict:
                        if key in candidate.keys() and candidate[key] == value:
                            next_values.append(value + "_present")
                            found = True
                            break;
                    
                    if not found:
                        next_values.append(value + "_absent")
            
        else:
            
            # For normal segments, get the value for the first segment
            for curr_dict in dicts:
                next_values.append(curr_dict[path[0]])
                
        # Recursively traverse the dictionaries down for the next segment
        if len(path) == 1:
            return next_values
        else:
            return self._getNextSegment(next_values, path[1:])
