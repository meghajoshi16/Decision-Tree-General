import csv 
import math 
import sys

#argv for this program when running for the first time, uncomment this section
# argv[1] = "small_train.csv"
# argv[2] = "small_test.csv"
# argv[3] = 4 
# argv[4] = "trainout_small.txt"  
# argv[5] = "testout_small.txt" 
# argv[6] = "metricsout.txt"

#Reformatting Data 
#with open(sys.argv[1]) as csvfile: 
with open(sys.argv[1]) as csvfile: 
    csv_doc = csv.reader(csvfile)
    csv_matrix = []  
    result_labels = []
    for row in csv_doc:
        csv_matrix.append(row[0:len(row)]) 

    attribute_names = csv_matrix[0]
    #print(attribute_names)
    column_names = attribute_names[0:len(attribute_names)-1] 
    #print(column_names)
    attributes = list(map(list, zip(*csv_matrix[1:])))
    #print(attributes)
    result_labels_train = attributes[len(attributes)-1]
    #print(result_labels)
    attributes_columns = attributes[:len(attributes)-1]
    #print(attributes_columns)
    label_names = list(set(result_labels)) 

#with open(sys.argv[1]) as csvfile_train:
with open(sys.argv[1]) as csvfile_train: 
    csv_doc_train = csv.reader(csvfile_train) 
    data_rows_train = []
    for row in csv_doc_train:
        data_rows_train.append(row)  
    cols_train = data_rows_train[0] 
    cols_train = cols_train[:-1]
    data_rows_train = data_rows_train[1:]  


with open(sys.argv[2]) as csvfile_test:
    csv_doc_test = csv.reader(csvfile_test) 
    data_rows_test = []
    for row in csv_doc_test:
        data_rows_test.append(row)  
    cols_test = data_rows_test[0] 
    cols_test = cols_test[:-1]
    data_rows_test = data_rows_test[1:] 
    result_labels_test = []
    for row in data_rows_test:  
        result_labels_test.append(row[len(row)-1])


    #Entropy of attribute columns 
    #input is a list 

    def entropy_of_data(resultLabels): #input is the list result_labels 
        label_names = list(set(resultLabels))
        label_1_count = 0
        label_2_count = 0
        label_1 = label_names[0]
        label_2 = label_names[1]

        for classif in resultLabels:
            if classif == label_1:
                label_1_count +=1 
            elif classif == label_2: 
                label_2_count +=1  
        total = label_1_count + label_2_count
        if label_1_count == 0: 
            part_1 = 0 
        elif label_1_count!=0: 
            part_1 = ((float(label_1_count)/total)*math.log((float(label_1_count)/total),2)) 
        if label_2_count == 0: 
            part_2 = 0
        elif label_2_count !=0: 
            part_2 = ((float(label_2_count)/total)*math.log((float(label_2_count)/total),2))

        entropy = -1*(part_1 + part_2 )
        return(entropy) 




    def mutual_information_attribute(attributeColumn, resultLabels, entropy):

        attribute_labels = list(set(attributeColumn)) #attribute_columns[0]/[1] will be fed in as attributeColumn
        #print(attribute_labels)
        att_label_1 = []
        att_label_2 = []
        for i, j in enumerate(attributeColumn):  #attribute_columns[0]/[1] will be fed in as attributeColumn
            if j == attribute_labels[0]:
                att_label_1.append(i) 
            elif j == attribute_labels[1]: 
                att_label_2.append(i)

        result_att_label1 = []
        result_att_label2 = [] 

        for i in att_label_1: 
            result_att_label1.append(resultLabels[i]) #should be resultLabels

        for i in att_label_2: 
            result_att_label2.append(resultLabels[i]) #should be resultLabels

        label_names = list(set(resultLabels)) #should be resultLabels
        #### This is for the number of positive, negative  per branch of attribute
        label_1_count_first = 0
        label_2_count_first = 0
        label_1_first = label_names[0]
        label_2_first = label_names[1]

        for classif in result_att_label1:
            if classif == label_1_first:
                label_1_count_first +=1 
            elif classif == label_2_first: 
                label_2_count_first +=1   

        label_1_count_sec = 0
        label_2_count_sec = 0
        label_1_sec = label_names[0]
        label_2_sec = label_names[1]

        for classif in result_att_label2:
            if classif == label_1_sec:
                label_1_count_sec +=1 
            elif classif == label_2_sec: 
                label_2_count_sec +=1   
     

        denom1 = label_1_count_first + label_2_count_first
        denom2 = label_1_count_sec + label_2_count_sec  

        final_denom = label_1_count_first + label_2_count_first + label_1_count_sec + label_2_count_sec  

        prob_first = float(denom1)/final_denom
        prob_sec =  float(denom2)/final_denom 


        if label_1_count_first==0 or denom1 == 0:
            first = 0
        else:       
            first = ((float(label_1_count_first)/denom1)*math.log((float(label_1_count_first)/denom1),2))


        if label_2_count_first==0 or denom1==0: 
            second = 0
            
        else:
            second = ((float(label_2_count_first)/denom1)*math.log((float(label_2_count_first)/denom1),2)) 


        if label_1_count_sec==0 or denom2==0:
            third = 0
        else:       
            third = ((float(label_1_count_sec)/denom2)*math.log((float(label_1_count_sec)/denom2),2))

    
        if label_2_count_sec==0 or denom2==0:
            fourth = 0
          
        else:       
            fourth = ((float(label_2_count_sec)/denom2)*math.log((float(label_2_count_sec)/denom2),2))

        conditional_entropy_1 = first + second 
        conditional_entropy_2 = third + fourth 

        conditional_entropy__total = prob_first*(conditional_entropy_1) + prob_sec*(conditional_entropy_2)
        mutual_information = entropy-(-1*(conditional_entropy__total))
        return(mutual_information) 



    def train_the_tree(attributeColumn, resultLabels, columnNames, entropy, max_depth, curr_depth = 0): 

        ### THE CASE IF MAX DEPTH == 0, JUST RETURN A TREE NODE WHERE TREE.DATA = MAJ CLASSIFIER ###

        if max_depth==0: 
            #Do majority vote classifier
            result_label_options = list(set(resultLabels)) 
            label1count = resultLabels.count(result_label_options[0])
            label2count = resultLabels.count(result_label_options[1]) 
            if label1count > label2count:  
                root = Tree(result_label_options[0])
            else: 
                root = Tree(result_label_options[1]) 
            return(root)  

        ###########################################################################################

        ### IF MAX_DEPTH > #OF ATTRIBUTES , MAKE THE MAX DEPTH THE NUMBER OF COLUMNS #### 

        if max_depth > len(attributes_columns): #(global var)
            max_depth = len(attributes_columns) 

        ############################################################################################ 
        if(max_depth==curr_depth):
            return(root)  

        ##### NOW CALCULATE THE MUTUAL INFORMATION SCORES ###### 
        mutual_information_scores = []
        for col in range(0,len(attributeColumn)):
            score = mutual_information_attribute(attributeColumn[col],resultLabels, entropy)
            mutual_information_scores.append(score) 
        attribute_to_split = max(mutual_information_scores)   


        ###### SPLITTING THE DATA FRAME BASED ON LABELS IN ATTRIBUTE COLUMN, AND EXTRACTING THE COUNTS FOR LEFT AND RIGHT  #####

        if attribute_to_split > 0: #Only split if mutual information > 0 
            curr_att_index = mutual_information_scores.index(attribute_to_split) #position of highest mutual information score
            
            attribute_to_split = columnNames[curr_att_index]  
             #After exxtracting the string of the current attribute, delete it 
            new_col_names_left = columnNames[:curr_att_index] + columnNames[curr_att_index+1:] 
            new_col_names_right = columnNames[:curr_att_index] + columnNames[curr_att_index+1:] 

             #The labels in the attribute we are splitting
            attribute_labels = list(set(attributeColumn[curr_att_index]))

            #Splitting first node by it's two labels  
            curr_att = attributeColumn[curr_att_index]
            right_side_indeces = []  
            left_side_indeces = []
            for i in range(0,len(curr_att)):
                if curr_att[i] == attribute_labels[0]: 
                    left_side_indeces.append(i) 
                elif curr_att[i] == attribute_labels[1]: 
                    right_side_indeces.append(i)  

            #Count the number of positive and negative values on left and right side 
            result_col_left = [resultLabels[i] for i in left_side_indeces] ### RESULT COLUMN FOR LEFT
            result_col_right = [resultLabels[j] for j in right_side_indeces] ### RESULT COLUMN FOR RIGHT
            result_label_options = list(set(resultLabels)) 
            positive_label = result_label_options[0] 
            negative_label = result_label_options[1] 
            #These are the counts of the result labels for each side
            left_side_pos_label_count = result_col_left.count(positive_label) #how many times the positive label is present in the left branch
            left_side_neg_label_count = result_col_left.count(negative_label) #how many times the negative label is present in the left branch
            right_side_pos_label_count = result_col_right.count(positive_label) #how many times the positive label is present in the right branch
            right_side_neg_label_count = result_col_right.count(negative_label) #how many times the negative label is present in the right branch   

            ### NEW COLUMNS FOR THE LEFT SIDE, MUST ALSO DELETE CURRENT ATTRIBUTE ###
            left_side_attribute_columns = [] 
            for col in  attributeColumn:
                subsetted_col = [col[i] for i in left_side_indeces] 
                left_side_attribute_columns.append(subsetted_col)  

            curr_left_side_attribute_columns = left_side_attribute_columns[:curr_att_index] + left_side_attribute_columns[curr_att_index+1:]  

            ### NEW COLUMNS FOR THE RIGHT SIDE, MUST ALSO DELETE CURRENT ATTRIBUTE ###

            right_side_attribute_columns = [] 
            for col in  attributeColumn:
                subsetted_col = [col[i] for i in right_side_indeces] 
                right_side_attribute_columns.append(subsetted_col) 

            curr_right_side_attribute_columns = right_side_attribute_columns[:curr_att_index] + right_side_attribute_columns[curr_att_index+1:] 

            curr_depth += 1  ### AT THIS POINT ATTRIBUTE HAS BEEN SPLIT, SO CAN ADD 1 TO DEPTH

            string_left = "| " * curr_depth + attribute_to_split + " = " + attribute_labels[0] + " : " + "[" + positive_label +" "+  str(left_side_pos_label_count) +" /"+ negative_label + " "+ str(left_side_neg_label_count) + "]"

            string_right = "| " * curr_depth + attribute_to_split + " = " + attribute_labels[1] + " : " + "[" + positive_label + " "+ str(right_side_pos_label_count) +" /"+ negative_label + " "+ str(right_side_neg_label_count) + "]"


            root = Tree(attribute_to_split) ###Trained on 1 stump  

        
            if(curr_depth == max_depth): 
                print(string_left)  
                if(left_side_pos_label_count >= left_side_neg_label_count): 
                    majL = positive_label 
                elif(left_side_pos_label_count < left_side_neg_label_count):
                    majL = negative_label
                root.left = Tree(majL)  
                root.left.label = attribute_labels[0]

                print(string_right)
                if(right_side_pos_label_count >= right_side_neg_label_count): 
                    majR = positive_label 
                elif(right_side_pos_label_count < right_side_neg_label_count):
                    majR = negative_label 
                root.right = Tree(majR)
                root.right.label = attribute_labels[1]  



            if((curr_depth!=max_depth) and (left_side_pos_label_count==0 or left_side_neg_label_count==0)): 
                print(string_left)  
                if(left_side_pos_label_count==0): 
                    majL = negative_label 
                elif(left_side_neg_label_count==0): 
                    majL = positive_label
                root.left = Tree(majL) 
                root.left.label = attribute_labels[0] 
            elif(curr_depth!=max_depth and left_side_pos_label_count!=0 and left_side_neg_label_count!=0):  
                print(string_left) 
                root.left = train_the_tree(curr_left_side_attribute_columns, result_col_left, new_col_names_left, entropy_of_data(result_col_left), max_depth, curr_depth)
                root.left.label = attribute_labels[0]  



            if((curr_depth!=max_depth) and (right_side_pos_label_count==0 or right_side_neg_label_count==0)): 
                print(string_right)  
                if(right_side_pos_label_count==0): 
                    majR = negative_label 
                elif(right_side_neg_label_count==0): 
                    majR = positive_label
                root.right = Tree(majR)
                root.right.label = attribute_labels[1] 
            elif(curr_depth!=max_depth and right_side_pos_label_count!=0 and right_side_neg_label_count!=0):  
                print(string_right)  
                root.right = train_the_tree(curr_right_side_attribute_columns, result_col_right, new_col_names_right, entropy_of_data(result_col_right), max_depth, curr_depth)
                root.right.label = attribute_labels[1] 

        return(root)

    def search(root, resultLabels, data_row, columnNames): 
        node_name = root.data  
        result_label_options = list(set(resultLabels)) 
        if node_name in result_label_options: 
            return node_name 
        index_of_att = columnNames.index(node_name) 
        label = data_row[index_of_att]  
        if root.left.label == label: 
            return(search(root.left, resultLabels, data_row, columnNames)) 
        if root.right.label == label: 
            return(search(root.right, resultLabels, data_row, columnNames))   

    def calculate_errors(root, resultLabels_train, data_rows_train, resultLabels_test,data_rows_test, columnNames):
        #argv[4] = train out 
        #argv[5] = test out  
        train_out = open(sys.argv[4],'w')
        test_out = open(sys.argv[5],'w')

        train_class = [] 
        for row in data_rows_train: 
            label_train = search(root, resultLabels_train, row, columnNames) 
            train_class.append(label_train)  
            train_out.write(str(label_train) + "\n")
        train_out.close()
        train_score = 0  
        for i in range(0, len(train_class)): 
            if train_class[i] != resultLabels_train[i]: 
                train_score += 1  
        training_error = float(train_score)/len(train_class) 
        training_error = round(training_error,6)


        test_class = []  
        for row in data_rows_test: 
            label_test = search(root, resultLabels_test, row, columnNames) 
            test_class.append(label_test)  
            test_out.write(str(label_test) + "\n")
        test_out.close()
        test_score = 0 
        for i in range(0,len(test_class)): 
            if test_class[i] != resultLabels_test[i]: 
                test_score += 1 
        testing_error = float(test_score)/len(test_class)  
        testing_error = round(testing_error,6)

        error_train_string = "error(train): " + str(training_error) 
        error_test_string = "error(test): " + str(testing_error) 
        
        metric = open(sys.argv[6],'w')
        metric.write(error_train_string + "\n")  
        metric.write(error_test_string)
        metric.close()

        return(training_error, testing_error)


            
class Tree(object): 
    def __init__(self, data): 
        self.left = None 
        self.right = None 
        self.data = data 
        self.label = None


d =train_the_tree(attributes_columns,result_labels_train, column_names, entropy_of_data(result_labels_train), int(sys.argv[3]))
calculate_errors(d, result_labels_train, data_rows_train, result_labels_test, data_rows_test,column_names)

        




    
