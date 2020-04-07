#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Author: Alexander Ulery
    
"""


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas
import pickle
import csv


"""
   freq, cond_p, inverse_p and Bayesian were largely influenced by the code
   written by Dr. C. Chan within Bayes_Model.py.
"""

def freq(x, opt='DF'):
    
    if opt != 'DF':
        if opt == 'dict':
            return { i: x.value_counts()[i] for i in x.unique()}
        else:
            return (x.name, {i: x.value_counts()[i] for i in x.unique()})
        
    return pandas.DataFrame([x.value_counts()[i] for i in x.unique()], index=x.unique(), columns=[x.name])


def cond_p(df, c, d):
    
    C = df.groupby(c).groups
    D = df.groupby(d).groups
    P_DC = { (i,j): (C[i] & D[j]).size / C[i].size
                for i in C.keys() for j in D.keys()}
    
    return P_DC


def inverse_p(df, cond_list, decision_list):
    
    p_list = [cond_p(df, decision_list, i) for i in cond_list]
    return p_list
    

def Bayes_model(csvDATA):
    cond_list = csvDATA.columns[:-1]
    decision_list = csvDATA.columns[-1]
    
    d_prior = freq(csvDATA[decision_list], 'dict')
    c_list = inverse_p(csvDATA, cond_list, decision_list)
    
    return (d_prior, c_list, cond_list, decision_list)  # return (dict, list, list, list)


#Inference engine to predict data output from classifier model
def predict(loaded_module, training_data):
    predicted_list = []
    
    with open(training_data, newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    choices = list(loaded_module[0].keys())   #choices from loaded_modules is a dict, this will make it an indexable list of keys
    key = 0
    for index in data:
        #print(index)
        for i in range(len(loaded_module[2])):   #data[index][len(index[1])-1],data[index][i]
            top_weight = 0
            p = 0
            for j in loaded_module[0]:
                weight = 1
                #X = int(data[key][len(index)-1])
                y = int(data[key][i])
                #print(loaded_module[2][i] + ' = ', loaded_module[1][i].get((X,y)))        
                weight = weight * loaded_module[1][i].get((j,y))
                if(weight > top_weight):
                    prediction = choices[p]
                    top_weight = weight
                p = p + 1
        predicted_list.append(prediction)
        key = key + 1

    return(predicted_list)


def menu1():
    
    print("\n(Note: A valid CSV file contains headers (attribute names) and training examples)")
    filename = input('\n    Please enter a filename of a valid CSV file (i.e. weather.csv): ')
    
    if(not(filename[len(filename) - 4:] == '.csv')):
        filename = filename + '.csv'
        
    csvDATA = pandas.read_csv(filename)
    d_prior, c_list, cond_list, decision_list = Bayes_model(csvDATA)
    
    print("\n    Classifier model successfully generated. Select option 2 in the main menu to save it.\n")
    return(d_prior, c_list, cond_list, decision_list)
    
    
def menu2(d_prior, c_list, cond_list, decision_list):
    
    savename = input("\nPlease enter a filename to save the current learned model(without extension): ")
    savename = savename + '.bin'
    
    pickle.dump((d_prior, c_list, cond_list, decision_list), open(savename, 'wb'))
    
    print("\n    Model successfully saved as " + savename + ". Try option 3 within the main menu to test it!\n")
    
    
def menu3():
    loadname = input("\nPlease enter the name of a previously saved model file: ")
    loadname = loadname + '.bin'
    
    #load classifier model
    loaded_model = pickle.load(open(loadname, 'rb'))
    print('\n    ', loadname, ' successfully loaded.')
    
    training_data = input("\nNow, enter the filename of testing data (in CSV form without headers): ")
    training_data = training_data + '.csv'
    
    predicted_list = predict(loaded_model, training_data)
    actual_list = []
    
    #open up training file to weigh against classifier module
    with open(training_data, 'r') as file:
        reader = csv.reader(file)
        for lines in reader:
            actual_list.append(int(lines[len(lines)-1]))
    
    
    matrix = confusion_matrix(actual_list, predicted_list)
    
    #output confusion matrix and accuracy score using sklearn metrics module
    print("\n  Confusion Matrix:\n")
    print(matrix)
    print("\n  ACCURACY SCORE:", accuracy_score(actual_list, predicted_list), '/ 1.0\n')
        
    
def menu4():
    MATRIX = []
    key = 0
    
    #keep running and building up a case matrix to classify until the 'Quit' option is chosen
    while True:
        opt = 0
        print("\n    4.1: Enter a new case interactively.\n")
        print("    4.2: Quit.\n")
        
        opt = input('Option: ')
        if(not (opt == '4.1' or opt == '4.2')):
            print("\n\n    ERROR: Please select a valid option (4.1 or 4.2)\n\n")
    
        if(opt == '4.1'):
            
            print("\n    Enter your case in a CSV format (i.e. 'yes','wet','cold',1)")
            
            if(key == 0):
                print("\n\n    *For your first entry, please enter attribute tags.")
                attribs = list(input("Attribute tags: "))
                for i in attribs:
                    if(i == ','):
                        attribs.remove(i)
                key += 1
                
            attrib_input = list(input("Condition attribute case: "))
            
            for i in attrib_input:
                if(i == ','):
                    attrib_input.remove(i)
            
            MATRIX.append(attrib_input)

            print("\nCurrent Case Queue:")
            print('    ', attribs)
            for i in MATRIX:
                print('    ', i)
            
        elif(opt == '4.2'):
            print("\n    Would you like to save your current model before quitting?")
            opt = input("    Say 'Y' if so, otherwise hit enter: ")
            
            if(opt == 'Y' or opt == 'y' or opt == 'YES' or opt == 'yes' or opt == 'Yes'):
                csvDATA = pandas.DataFrame(MATRIX, columns = attribs)
                d_prior, c_list, cond_list, decision_list = Bayes_model(csvDATA)
                menu2(d_prior, c_list, cond_list, decision_list)
                
            break
    
class py_nb():

    print("\nAuthor: Alexander Ulery \n")
    print("Please select an option from the menu below.\n")
    
    while True:
        
        option = 0
        
        print("\n MAIN MENU: \n")
        print("1: Learn a Naive Bayesian classifier from categorical data.\n")
        print("2: Save a model.\n")
        print("3: Load a model and test its accuracy.\n")
        print("4: Apply a Naive Bayesian classifier to new cases interactively.\n")
        print("5: Quit.\n")
        
        option = int(input('Option: '))
        if (not (1 <= option <= 5)):
            print("\n\nERROR: Please select a valid option (1, 2, 3, 4, 5)\n\n")
        
        if(option == 1):
            d_prior, c_list, cond_list, decision_list = menu1()
        elif(option == 2):
            menu2(d_prior, c_list, cond_list, decision_list)
        elif(option == 3):
            menu3()
        elif(option == 4):
            menu4()
        elif(option == 5):
            break
   
    
py_nb()