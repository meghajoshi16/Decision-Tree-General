import csv 
import math
import sys
#sys.argv[0] 
#csv_string = sys.argv[1]
with open(sys.argv[1]) as csvfile:
    csv_doc = csv.reader(csvfile)
    label = []
    for row in csv_doc:
        label.append(row[len(row)-1])
label_1=list(set(label[1:]))[0]
label_2=list(set(label[1:]))[1] 
label_1_count = 0
label_2_count = 0
for classif in label[1:]:
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
entropy = str(round(entropy,12))
entropy_string =  "entropy: " + str(entropy)

#percent of incorrectly classified
error = min(float(label_1_count),float(label_2_count))/float(total)
error = str(round(error,12))
error_string = "error: " + str(error) 

f = open(sys.argv[2],'w')
f.write(entropy_string + "\n")  
f.write(error_string)
f.close()




    



     

