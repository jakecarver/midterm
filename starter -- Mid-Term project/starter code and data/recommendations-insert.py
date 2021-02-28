'''
Calculate User-based and Item-based CF Recommendations using ML100K dataset

<< Insert the code below into your recommendations.py >>

'''

import os

################################################################################
## This def is here only to prevent a Python error message when running
## this code. Your code should already have a completed version of this function,
## so you don't need to insert this def into your recommendations.py!!
def from_file_to_dict(path, datafile, itemfile):
    return {}
################################################################################


################################################################################
## Bug ALERT fixes

## In def sim_distance(prefs,person1,person2):
# Remove or comment out this code, it was just for demo. Not a big deal for 
#    samll datasets but can impact larger datasets. Use list comprehension 
#    code instead  ..
#
# sum_of_squares = 0
# for item in prefs[person1]:
#    if item in prefs[person2]:
#        #print(item, prefs[person1][item], prefs[person2][item])
#        sq = pow(prefs[person1][item]-prefs[person2][item],2)
#        #print (sq)
#        sum_of_squares += sq

## In def calculateSimilarItems(prefs,n=10,similarity=sim_pearson):
# change ...  n=10   
# to ...      n=100 ## not really a bug, but we'll want this for larger datasets
#
# change ...  print ("%d / %d") % (c,len(itemPrefs)) ## this is a bug!
# to ...      print ("%d / %d" % (c,len(itemPrefs)))

## in def from_file_to_dict(path, datafile, itemfile):
# You may need to add an encoding parameter in the with statement to read the 
# ml-100k dataset ..
#    with open (path + '/' + itemfile, encoding='iso8859') as myfile:

## Also, you will want to revisit some of your code to make sure it can handle
## printing (or not printing) hundreds or thousands of users and items!

################################################################################


def main():
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    done = False
    prefs = {}
    
    while not done:
        print()

        ########################################################################
        ########################################################################   
        
        ## add this new command to the list of commands in the user interface
        file_io = input('RML(ead ml100K data)?, ') 
        
        ## add this new command to the list of commands as an elif
        if file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )
            
        ########################################################################
        ########################################################################        
         
        else:
            done = True
            
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()
    